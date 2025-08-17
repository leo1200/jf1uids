# general
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, lax
import time

# typing
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from typing import Tuple, Union

# jf1uids classes
from jf1uids.option_classes.simulation_config import SimulationConfig, STATE_TYPE
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids._physics_modules._self_gravity._poisson_solver import _compute_gravitational_potential
from jf1uids._physics_modules._self_gravity._self_gravity import _gravitational_source_term_along_axis

from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, primitive_state_from_conserved

class PointMassPotential:
    """
    Point-mass gravitational potential for two-body interactions.
    """
    def __init__(self, M1=1.0, M2=1.0):
        self.M1 = M1
        self.M2 = M2

    def acceleration(self, pos1, pos2):
        diff = pos1 - pos2
        inv_r2 = 1.0 / jnp.dot(diff, diff)
        inv_r = jnp.sqrt(inv_r2)
        return -self.M2 * (inv_r2 * inv_r) * diff
    
    def potential(self,x,y,z,x2,y2,z2):
        """ given position, return potential """
        dx=(x-x2)
        dy=(y-y2)
        dz=(z-z2)
        return -(self.M1*self.M2) / jnp.sqrt(dx**2 + dy**2 + dz**2)

@partial(jit, static_argnums=(2,))
def rk4_step_full(state, h, potential):
    """
    One RK4 step for the full two-body system.
    state: array [t1,x1,y1,z1,vx1,vy1,vz1, t2,x2,y2,z2,vx2,vy2,vz2]
    Returns new_state of same shape.
    """
    hh = 0.5 * h
    h6 = h / 6.0

    def deriv(s):
        txv1 = s[:7]
        txv2 = s[7:14]
        # time derivative
        dt = 1.0
        # positions and velocities
        vel1 = txv1[4:7]
        vel2 = txv2[4:7]
        # accelerations
        acc1 = potential.acceleration(txv1[1:4], txv2[1:4])
        acc2 = -acc1 * (potential.M1 / potential.M2)
        return jnp.concatenate([jnp.array([dt]), vel1, acc1,
                                jnp.array([dt]), vel2, acc2])

    k1 = deriv(state)
    k2 = deriv(state + hh * k1)
    k3 = deriv(state + hh * k2)
    k4 = deriv(state + h   * k3)

    return state + h6 * (k1 + 2*(k2 + k3) + k4)

### Nearest Grid Point (NGP) particle deposition (might not be good with FFT poisson solver)
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['grid_shape', 'grid_spacing'])
def _deposit_particles_ngp(
    particle_positions: Float[Array, "n 3"],
    particle_masses:    Float[Array, "n"],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[Array, "nx ny nz"]:
    """
    Deposit n point-masses to nearest grid cell (NGP).
    Positions in same units as grid, origin at (0,0,0).
    """
    grid_extent = jnp.array(grid_shape) * grid_spacing    # e.g. [Nx*dx, Ny*dy, Nz*dz]
    grid_min    = -0.5 * grid_extent                      
    # map world->grid indices by subtracting the minimum corner:
    idx = ((particle_positions - grid_min) // grid_spacing).astype(int)
    idx = jnp.clip(idx, 0, jnp.array(grid_shape) - 1)
    # Flatten grid and scatter-add masses
    flat_idx = idx[:,0] * (grid_shape[1]*grid_shape[2]) + idx[:,1] * grid_shape[2] + idx[:,2]
    rho_flat = jnp.zeros(grid_shape[0]*grid_shape[1]*grid_shape[2])
    rho_flat = rho_flat.at[flat_idx].add(particle_masses)
    return rho_flat.reshape(grid_shape)

### Cloud-In-Cell (CIC) particle deposition (might be better?)
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=('grid_shape', 'grid_spacing'))
def _deposit_particles_cic(
    particle_positions: Float[jnp.ndarray, "n 3"],
    particle_masses:    Float[jnp.ndarray, "n"],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[jnp.ndarray, "nx ny nz"]:
    """
    Cloud-In-Cell (CIC) deposit (3D).

    particle_positions: (N,3) world coords
    particle_masses:    (N,)
    grid_shape:         (nx,ny,nz) -- MUST be a Python tuple at call time (static)
    grid_spacing:       scalar dx (treated as static here)
    """
    nx, ny, nz = grid_shape
    grid_extent = jnp.array([nx, ny, nz]) * grid_spacing
    grid_min = -0.5 * grid_extent
    # relative continuous index in grid coordinates
    rel = (particle_positions - grid_min) / grid_spacing   # (N,3)
    i0 = jnp.floor(rel).astype(jnp.int32)                 # lower index (N,3)
    f  = rel - i0.astype(rel.dtype)                       # fractional part (N,3) in [0,1)
    # 8 neighbor offsets for CIC (cartesian product of {0,1}^3)
    offsets = jnp.array([
        [0,0,0],[0,0,1],[0,1,0],[0,1,1],
        [1,0,0],[1,0,1],[1,1,0],[1,1,1],
    ], dtype=jnp.int32)                                   # (8,3)

    # neighbor indices (N,8,3)
    neigh_idx = i0[:, None, :] + offsets[None, :, :]
    # clip indices to grid boundaries (non-periodic)
    max_idx = jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32)
    neigh_idx = jnp.clip(neigh_idx, 0, max_idx)
    # weights: for each dim weight is (1-f) if offset==0 else f; multiply over dims -> (N,8)
    f_b = f[:, None, :]                                   # (N,1,3)
    # boolean mask of offsets==0 broadcasted -> choose (1-f) or f
    w_comp = jnp.where(offsets[None, :, :] == 0, 1.0 - f_b, f_b)  # (N,8,3)
    weights = jnp.prod(w_comp, axis=-1)                   # (N,8)
    # linearize 3D indices to flat indices (row-major: x*(ny*nz) + y*nz + z)
    flat_idx = (neigh_idx[..., 0] * (ny * nz)
                + neigh_idx[..., 1] * nz
                + neigh_idx[..., 2])                      # (N,8)
    # flatten for scatter
    flat_idx_flat = flat_idx.reshape(-1)                  # (N*8,)
    values_flat = (particle_masses[:, None] * weights).reshape(-1)  # (N*8,)
    n_cells = nx * ny * nz                                # Python int (static)
    rho_flat = jnp.zeros(n_cells, dtype=particle_masses.dtype)
    rho_flat = rho_flat.at[flat_idx_flat].add(values_flat)

    return rho_flat.reshape((nx, ny, nz))

# Triangular-Shaped-Cloud (TSC) particle deposition (quadratic B-spline)
# This is a more accurate method than CIC, but more expensive. It spreads each particle’s mass to the
# 3 nearest grid points along each axis (3×3×3 = 27 cells in 3D) with quadratic weights.
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=('grid_shape', 'grid_spacing'))
def _deposit_particles_tsc(
    particle_positions: Float[jnp.ndarray, "n 3"],
    particle_masses:    Float[jnp.ndarray, "n"],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[jnp.ndarray, "nx ny nz"]:
    """
    TSC (Triangular-Shaped-Cloud) deposit in 3D.

    - particle_positions: (N,3) world coordinates
    - particle_masses:    (N,)
    - grid_shape:         (nx,ny,nz) as a Python tuple (must be static at call time)
    - grid_spacing:       scalar dx (treated as static here)
    Returns: rho (nx,ny,nz) with total mass conserved.
    """
    nx, ny, nz = grid_shape
    grid_extent = jnp.array([nx, ny, nz]) * grid_spacing
    grid_min = -0.5 * grid_extent   # center the grid
    # continuous position in grid units (index-space)
    rel = (particle_positions - grid_min) / grid_spacing    # (N,3)
    # floor(rel) gives a central index; neighbors are floor(rel)-1, floor(rel), floor(rel)+1
    i_center = jnp.floor(rel).astype(jnp.int32)             # (N,3)
    # Offsets for TSC: cartesian product of [-1,0,1]^3 -> 27 neighbors
    offsets = jnp.array([[i, j, k]
                         for i in (-1, 0, 1)
                         for j in (-1, 0, 1)
                         for k in (-1, 0, 1)], dtype=jnp.int32)   # (27,3)
    # neighbor indices (N,27,3)
    neigh_idx = i_center[:, None, :] + offsets[None, :, :]   # (N,27,3)
    # Clip indices to grid bounds (non-periodic behaviour)
    max_idx = jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32)
    neigh_idx = jnp.clip(neigh_idx, 0, max_idx)
    # compute 1D distances 
    r = rel[:, None, :] - neigh_idx.astype(rel.dtype)       # (N,27,3)
    s = jnp.abs(r)                                          # (N,27,3)
    # 1D TSC kernel evaluated vectorized:
    def W1D_from_s(s_component):
        w = jnp.where(s_component <= 0.5,
                      0.75 - s_component**2,
                      jnp.where(s_component <= 1.5,
                                0.5 * (1.5 - s_component)**2,
                                0.0))
        return w

    wx = W1D_from_s(s[..., 0])   # (N,27)
    wy = W1D_from_s(s[..., 1])   # (N,27)
    wz = W1D_from_s(s[..., 2])   # (N,27)
    weights = wx * wy * wz      # (N,27)
    # Flatten neighbor flat indices and weighted mass values for scatter
    flat_idx = (neigh_idx[..., 0] * (ny * nz)
                + neigh_idx[..., 1] * nz
                + neigh_idx[..., 2])                    # (N,27)

    flat_idx_flat = flat_idx.reshape(-1)                 # (N*27,)
    values_flat = (particle_masses[:, None] * weights).reshape(-1)  # (N*27,)
    n_cells = nx * ny * nz    # Python int (static)
    rho_flat = jnp.zeros(n_cells, dtype=particle_masses.dtype)
    rho_flat = rho_flat.at[flat_idx_flat].add(values_flat)

    return rho_flat.reshape((nx, ny, nz))


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def binary_full(
    primitive_state: STATE_TYPE,
    old_primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    dt: Float[Array, ""],
    # binary_masses: Union[None, Float[Array, "2"]] = None,
    binary_state: Union[None, Float[Array, "14"]] = None
    ) -> Tuple[STATE_TYPE, Float[Array, "14"]]:  
    
    rho_gas = old_primitive_state[registered_variables.density_index]
    M1 = params.binary_params.masses[0]
    M2 = params.binary_params.masses[1]
    binary_state = binary_state
    state_flat = binary_state.reshape(-1)            # now shape (14,) (if not already flattened)
    new_flat   = rk4_step_full(state_flat, dt, PointMassPotential(M1, M2))   
    pos1 = new_flat[1:4]
    pos2 = new_flat[8:11]
    particle_positions = jnp.stack([pos1, pos2], axis=0)
    particle_masses    = jnp.array([M1, M2])
    grid_shape   = rho_gas.shape
    grid_spacing = config.grid_spacing
    if config.binary_config.deposit_particles == "ngp":
        rho_part     = _deposit_particles_ngp(particle_positions, particle_masses, grid_shape, grid_spacing)
    elif config.binary_config.deposit_particles == "cic":
        rho_part     = _deposit_particles_cic(particle_positions, particle_masses, grid_shape, grid_spacing)
    elif config.binary_config.deposit_particles == "tsc":
        rho_part     = _deposit_particles_tsc(particle_positions, particle_masses, grid_shape, grid_spacing)
    else:
        raise ValueError(f"Unknown deposit_particles method: {config.binary_config.deposit_particles}")
    rho_tot = rho_gas + rho_part
    # compute potential from combined density
    potential = _compute_gravitational_potential(rho_tot, config.grid_spacing, config, gravitational_constant)

    source_term = jnp.zeros_like(primitive_state)

    for i in range(config.dimensionality):
        source_term = source_term + _gravitational_source_term_along_axis(
                                        potential,
                                        old_primitive_state,
                                        config.grid_spacing,
                                        registered_variables,
                                        dt,
                                        gamma,
                                        config,
                                        params,
                                        helper_data,
                                        i + 1
                                    )

    conserved_state = conserved_state_from_primitive(primitive_state, gamma, config, registered_variables)

    conserved_state = conserved_state + dt * source_term

    primitive_state = primitive_state_from_conserved(conserved_state, gamma, config, registered_variables)

    primitive_state = _boundary_handler(primitive_state, config)

    return primitive_state, new_flat 

def integrateBinary(orbit1, orbit2, M1, M2, h, T):
    txv1 = orbit1     
    txv2 = orbit2  
    state0 = jnp.concatenate([txv1, txv2])
    M1 = M1
    M2 = M2
    pot = PointMassPotential(M1, M2)
    h = h
    T = T
    num_steps = int(jnp.ceil(T / h))

    @jit
    def run_with_fori_loop(state0):
        def body_fn(i, carry):
            state, traj = carry
            new_state = rk4_step_full(state, h, pot)
            traj = traj.at[i].set(new_state)
            return new_state, traj

        traj = jnp.zeros((num_steps, state0.shape[0]), dtype=state0.dtype)
        _, traj = lax.fori_loop(0, num_steps, body_fn, (state0, traj))
        return traj[-1], traj
    
    t0 = time.perf_counter()
    final_state, traj = run_with_fori_loop(state0)
    t1 = time.perf_counter()
    print(f"RK4 took {t1 - t0:.4f} seconds")

    """
    def scan_body(s, _):
        s_new = rk4_step_full(s, h, pot)
        return s_new, s_new

    jit_scan = jit(lambda s: lax.scan(scan_body, s, None, length=num_steps))
    _ = jit_scan(state0)
    final_state, traj = jit_scan(state0)
    """
    # trajectories
    traj = jnp.stack(traj)  
    traj1 = traj[:, :7]
    traj2 = traj[:, 7:14]

    totalM = M1 + M2
    # Center-of-mass frame
    Rx = (M2 * traj2[:,1] + M1 * traj1[:,1]) / totalM
    Ry = (M2 * traj2[:,2] + M1 * traj1[:,2]) / totalM
    Rz = (M2 * traj2[:,3] + M1 * traj1[:,3]) / totalM
    traj1_com = traj1.at[:,1].set(traj1[:,1] - Rx)
    traj1_com = traj1_com.at[:,2].set(traj1[:,2] - Ry)
    traj1_com = traj1_com.at[:,3].set(traj1[:,3] - Rz)
    traj2_com = traj2.at[:,1].set(traj2[:,1] - Rx)
    traj2_com = traj2_com.at[:,2].set(traj2[:,2] - Ry)
    traj2_com = traj2_com.at[:,3].set(traj2[:,3] - Rz)

    return traj1_com, traj2_com

def quantity_test(orbit1, orbit2, M1, M2, h, T):
    traj1, traj2 = integrateBinary(orbit1, orbit2, M1, M2, h, T)
    vx=(traj1[:,4]-traj2[:,4])
    vy=(traj1[:,5]-traj2[:,5])
    x_rel=(traj1[:,1]-traj2[:,1])
    y_rel=(traj1[:,2]-traj2[:,2])
    E   = 0.5*(M1*(traj1[:,4]**2+traj1[:,5]**2) + M2*(traj2[:,4]**2+traj2[:,5]**2)) \
    + PointMassPotential(M1,M2).potential(traj1[:,1],traj1[:,2],traj1[:,3],traj2[:,1],traj2[:,2],traj2[:,3])

    dE  = (E-E[0])/E[0]
    L = x_rel*vy - y_rel*vx    # x*vy - y*vx
    dL  = (L-L[0])/L[0]
    return traj1, traj2, dE, dL

if __name__ == "__main__":
    orbit1 = jnp.array([0.0,  0.3, 0.0, 0.0, 0.0, 0.5, 0.0])
    orbit2 = jnp.array([0.0, -0.7, 0.0, 0.0, 0.0,-0.4, 0.0])
    M1 = 0.5
    M2 = 0.8
    h = 0.0008
    T = 20
    traj1_com, traj2_com, dE, dL = quantity_test(orbit1, orbit2, M1, M2, h, T)
    # traj1_com, traj2_com = integrate(orbit1, orbit2, M1, M2, h, T)
    Rx = (M2 * traj2_com[0,1] + M1 * traj1_com[0,1]) / (M1 + M2)
    Ry = (M2 * traj2_com[0,2] + M1 * traj1_com[0,2]) / (M1 + M2)
    from matplotlib import pyplot as plt
    import os
    save_path = os.path.join(
    '..','..', '..', 'tests', 'binary', 'RK4_orbits'
    )
    plt.plot(traj1_com[:,1],traj1_com[:,2],markersize=.1,linewidth=1) 
    plt.plot(traj2_com[:,1],traj2_com[:,2],markersize=.1,color="r",linewidth=1) 
    plt.scatter(Rx, Ry, color='black', s=10)  # Center of mass
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.title("Binary Orbits")
    plt.savefig(save_path+"/JAX_orbits3.png", dpi=300)
    plt.close()
    # energy / angular momentum errors
    plt.plot(traj1_com[:,0],dE,label=r'$\Delta E/E_0$')
    plt.plot(traj1_com[:,0],dL,label=r'$\Delta L/L_0$')
    plt.xlabel('time')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.savefig(save_path+"/JAX_orbits_errors3.png", dpi=300)
