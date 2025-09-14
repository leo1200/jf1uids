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

def acceleration(positions: jnp.ndarray, masses: jnp.ndarray, eps: float = 1e-12):
    """
    Compute accelerations for n bodies due to mutual gravity.
    positions: shape (n,3)
    masses: shape (n,)
    returns: acc shape (n,3) where acc[i] = sum_{j != i} - masses[j] * (r_i - r_j) / |r_i - r_j|^3
    (G is assumed 1, consistent with original code's unit treatment)
    """
            # positions[:, None, :] - positions[None, :, :] -> shape (n,n,3) where diff[i,j] = r_i - r_j
    diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
    r2 = jnp.sum(diff ** 2, axis=-1)  # (n, n)
    # Avoid dividing by zero on the diagonal by adding small eps
    inv_r3 = jnp.where(r2 > 0, 1.0 / (r2 * jnp.sqrt(r2) + eps), 0.0)  # (n, n)
    # mass-weighted factor for each pair (broadcast masses[j] over i)
    mass_factors = masses[None, :]  # (1, n)
    # acceleration contribution from j on i: - mass_j * diff_ij * inv_r3_ij
    contrib = - (mass_factors[..., None] * diff) * inv_r3[..., None]  # (n, n, 3)
    # zero out self-contrib (diagonal) in case numerical nonzero
    contrib = contrib * (1.0 - jnp.eye(positions.shape[0])[:, :, None])
    # sum over j
    acc = jnp.sum(contrib, axis=1)  # (n, 3)
    return acc

@jit 
def rk4_step_nbody(state: jnp.ndarray, h: float, masses: jnp.ndarray):
    """
    One RK4 step for n-body system.
    state: array shape (n,7) where each row is [t, x, y, z, vx, vy, vz]
    masses: array shape (n,)
    returns: new_state shape (n,7)
    """
    hh = 0.5 * h
    h6 = h / 6.0

    def deriv(s):
        # s shape (n,7)
        n = s.shape[0]
        dt_col = jnp.ones((n, 1), dtype=s.dtype)  # time derivative (1 per body)
        positions = s[:, 1:4]  # (n,3)
        velocities = s[:, 4:7]  # (n,3)
        acc = acceleration(positions, masses)  # (n,3)
        return jnp.concatenate([dt_col, velocities, acc], axis=1)  # (n,7)

    k1 = deriv(state)
    k2 = deriv(state + hh * k1)
    k3 = deriv(state + hh * k2)
    k4 = deriv(state + h * k3)

    return state + h6 * (k1 + 2.0 * (k2 + k3) + k4)


### Nearest Grid Point (NGP) particle deposition (might not be good with FFT poisson solver)
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['grid_shape', 'grid_spacing'])
def _deposit_particles_ngp(
    particle_positions: Float[Array, "n 3"],
    particle_masses:    Union[Float[Array, ""], Float[Array, "n"]],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[Array, "nx ny nz"]:
    """
    Deposit n point-masses to nearest grid cell (NGP).
    Positions in same units as grid, origin at (0,0,0).
    """
    grid_extent = jnp.array(grid_shape) * grid_spacing   
    grid_min    = -0.5 * grid_extent                  
    particle_densities = particle_masses / (grid_spacing ** 3)    
    # map world->grid indices by subtracting the minimum corner:
    idx = ((particle_positions - grid_min) // grid_spacing).astype(int)
    idx = jnp.clip(idx, 0, jnp.array(grid_shape) - 1)
    # Flatten grid and scatter-add masses
    flat_idx = idx[:,0] * (grid_shape[1]*grid_shape[2]) + idx[:,1] * grid_shape[2] + idx[:,2]
    rho_flat = jnp.zeros(grid_shape[0]*grid_shape[1]*grid_shape[2])
    rho_flat = rho_flat.at[flat_idx].add(particle_densities)
    return rho_flat.reshape(grid_shape)

### Cloud-In-Cell (CIC) particle deposition (might be better?)
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=('grid_shape', 'grid_spacing'))
def _deposit_particles_cic(
    particle_positions: Float[jnp.ndarray, "n 3"],
    particle_masses:    Union[Float[Array, ""], Float[Array, "n"]],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[jnp.ndarray, "nx ny nz"]:
    """
    Cloud-In-Cell (CIC) deposit (3D).
    """
    nx, ny, nz = grid_shape
    grid_extent = jnp.array([nx, ny, nz]) * grid_spacing
    grid_min = -0.5 * grid_extent
    particle_densities = particle_masses / (grid_spacing ** 3)  
    # relative index in grid coordinates
    rel = (particle_positions - grid_min) / grid_spacing  
    i0 = jnp.floor(rel).astype(jnp.int32)                 
    f  = rel - i0.astype(rel.dtype)                      
    # 8 neighbor offsets for CIC 
    offsets = jnp.array([
        [0,0,0],[0,0,1],[0,1,0],[0,1,1],
        [1,0,0],[1,0,1],[1,1,0],[1,1,1],
    ], dtype=jnp.int32)                                   # (8,3)

    neigh_idx = i0[:, None, :] + offsets[None, :, :]
    # clip indices to grid boundaries (non-periodic)
    max_idx = jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32)
    neigh_idx = jnp.clip(neigh_idx, 0, max_idx)
    # weights: for each dim weight is (1-f) if offset==0 else f; multiply over dims -> (N,8)
    f_b = f[:, None, :]                                   # (N,1,3)
    # boolean mask of offsets==0 broadcasted -> choose (1-f) or f
    w_comp = jnp.where(offsets[None, :, :] == 0, 1.0 - f_b, f_b)  
    weights = jnp.prod(w_comp, axis=-1)                  
    # linearize 3D indices to flat indices (row-major: x*(ny*nz) + y*nz + z)
    flat_idx = (neigh_idx[..., 0] * (ny * nz)
                + neigh_idx[..., 1] * nz
                + neigh_idx[..., 2])                      
    # flatten for scatter
    flat_idx_flat = flat_idx.reshape(-1)                  
    values_flat = (particle_densities[:, None] * weights).reshape(-1)  
    n_cells = nx * ny * nz                               
    rho_flat = jnp.zeros(n_cells, dtype=particle_densities.dtype)
    rho_flat = rho_flat.at[flat_idx_flat].add(values_flat)

    return rho_flat.reshape((nx, ny, nz))

# Triangular-Shaped-Cloud (TSC) particle deposition (quadratic B-spline)
# This is a more accurate method than CIC, but more expensive. It spreads each particle’s mass to the
# 3 nearest grid points along each axis (3×3×3 = 27 cells in 3D) with quadratic weights.
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=('grid_shape', 'grid_spacing'))
def _deposit_particles_tsc(
    particle_positions: Float[jnp.ndarray, "n 3"],
    particle_masses:    Union[Float[Array, ""], Float[Array, "n"]],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[jnp.ndarray, "nx ny nz"]:
    """
    TSC (Triangular-Shaped-Cloud) deposit in 3D
    """
    nx, ny, nz = grid_shape
    grid_extent = jnp.array([nx, ny, nz]) * grid_spacing
    grid_min = -0.5 * grid_extent   
    particle_densities = particle_masses / (grid_spacing ** 3) 
    # continuous position in grid units 
    rel = (particle_positions - grid_min) / grid_spacing   
    # floor(rel) gives a central index; neighbors are floor(rel)-1, floor(rel), floor(rel)+1
    i_center = jnp.floor(rel).astype(jnp.int32)           
    # Offsets for TSC: 27 neighbors
    offsets = jnp.array([[i, j, k]
                         for i in (-1, 0, 1)
                         for j in (-1, 0, 1)
                         for k in (-1, 0, 1)], dtype=jnp.int32)   
    # neighbor indices (N,27,3)
    neigh_idx = i_center[:, None, :] + offsets[None, :, :]   
    max_idx = jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32)
    neigh_idx = jnp.clip(neigh_idx, 0, max_idx)
    # 1D distances 
    r = rel[:, None, :] - neigh_idx.astype(rel.dtype)       
    s = jnp.abs(r)                                          
    # 1D TSC kernel evaluated vectorized:
    def W1D_from_s(s_component):
        w = jnp.where(s_component <= 0.5,
                      0.75 - s_component**2,
                      jnp.where(s_component <= 1.5,
                                0.5 * (1.5 - s_component)**2,
                                0.0))
        return w

    wx = W1D_from_s(s[..., 0])   
    wy = W1D_from_s(s[..., 1])   
    wz = W1D_from_s(s[..., 2])   
    weights = wx * wy * wz      
    # Flatten neighbor flat indices and weighted mass values for scatter
    flat_idx = (neigh_idx[..., 0] * (ny * nz)
                + neigh_idx[..., 1] * nz
                + neigh_idx[..., 2])                    

    flat_idx_flat = flat_idx.reshape(-1)                 
    values_flat = (particle_densities[:, None] * weights).reshape(-1)  
    n_cells = nx * ny * nz   
    rho_flat = jnp.zeros(n_cells, dtype=particle_densities.dtype)
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
    binary_state: Union[None, Float[Array, "n"]] = None
    ) -> Tuple[STATE_TYPE, Float[Array, "n"]]:  
    
    rho_gas = old_primitive_state[registered_variables.density_index]
    particle_masses = params.binary_params.masses
    if config.binary_config.central_object_only == False:
        if binary_state.ndim == 1:
            n_bodies = particle_masses.size
            state = binary_state.reshape((n_bodies, 7)) 
        else:
            state = binary_state         
        new_state = rk4_step_nbody(state, dt, particle_masses)  
        particle_positions = new_state[:, 1:4]
        new_state = new_state.reshape(-1)
    elif config.binary_config.central_object_only == True:
        particle_positions = jnp.zeros((1, config.dimensionality))
        new_state = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    grid_shape   = rho_gas.shape
    grid_spacing = config.grid_spacing
    if config.binary_config.deposit_particles == 0:
        rho_part = _deposit_particles_ngp(particle_positions, particle_masses, grid_shape, grid_spacing)
    elif config.binary_config.deposit_particles == 1:
        rho_part = _deposit_particles_cic(particle_positions, particle_masses, grid_shape, grid_spacing)
    elif config.binary_config.deposit_particles == 2:
        rho_part = _deposit_particles_tsc(particle_positions, particle_masses, grid_shape, grid_spacing)
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

    return primitive_state, new_state 


def integrate_nbody(orbits: jnp.ndarray, masses: jnp.ndarray, h: float, T: float, eps: float = 1e-12):
    """
    orbits: jnp.array shape (n,7) initial rows [t, x, y, z, vx, vy, vz]
    masses: jnp.array shape (n,)
    h: timestep
    T: total integration time
    returns: traj_com shape (num_steps, n, 7) positions/velocities in COM frame
    """
    state0 = orbits  # (n,7)
    num_steps = int(jnp.ceil(T / h))
    n = state0.shape[0]
    totalM = jnp.sum(masses)

    @jit
    def run_with_fori_loop(state0):
        def body_fn(i, carry):
            state, traj = carry  # state (n,7), traj (num_steps, n, 7)
            new_state = rk4_step_nbody(state, h, masses)
            traj = traj.at[i].set(new_state)
            return new_state, traj

        traj = jnp.zeros((num_steps, n, 7), dtype=state0.dtype)
        _, traj = lax.fori_loop(0, num_steps, body_fn, (state0, traj))
        return traj

    t0 = time.perf_counter()
    traj = run_with_fori_loop(state0)
    t1 = time.perf_counter()
    print(f"n-body RK4 took {t1 - t0:.4f} seconds")

    # Transform trajectories into center-of-mass frame (positions only; times/vels adjusted)
    # Compute center of mass position at every timestep: COM(t) = sum_i m_i * pos_i(t) / totalM
    positions = traj[:, :, 1:4]  # (num_steps, n, 3)
    # Broadcast masses to multiply across timesteps
    weighted = positions * masses[None, :, None]  # (num_steps, n, 3)
    COM = jnp.sum(weighted, axis=1) / totalM  # (num_steps, 3)
    # Subtract COM from each body's positions for all timesteps
    positions_com = positions - COM[:, None, :]  # (num_steps, n, 3)

    # Build traj_com by replacing positions with COM-subtracted positions
    traj_com = traj.at[:, :, 1:4].set(positions_com)

    return traj_com


def set_axes_equal(ax):
    """
    Make 3D axes have equal scale.
    Matplotlib 3D doesn't support `ax.set_aspect('equal')` for 3D
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_3d_trajectories(traj,
                         elev: float = 30.0,
                         azim: float = 45.0,
                         show_initial: bool = True,
                         figsize=(12, 6),
                         title="n-body trajectories (3D view)"):
    """
    Plot 3D trajectories. Parameters
    - traj: array-like (num_steps, n, 7) to plot (typically COM-frame)
    - elev, azim: camera elevation and azimuth (degrees) -> diagonal viewpoint
    - show_initial: if True mark initial positions with larger markers
    """

    traj = np.asarray(traj)  # convert JAX arrays if needed
    num_steps, n, _ = traj.shape

    fig = plt.figure(figsize=figsize)

    def _plot_on_axis(ax, data, subtitle):
        # data shape: (num_steps, n, 7)
        for i in range(n):
            xs = data[:, i, 1]  # x
            ys = data[:, i, 2]  # y
            zs = data[:, i, 3]  # z
            ax.plot(xs, ys, zs, linewidth=1, label=f'body {i}')
            if show_initial:
                ax.scatter(xs[0], ys[0], zs[0], s=40)  # initial point
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(subtitle)
        ax.view_init(elev=elev, azim=azim)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
        set_axes_equal(ax)

  
    ax = fig.add_subplot(111, projection='3d')
    _plot_on_axis(ax, traj, title)
    size = 5
    ax.set_xlim3d([-size, size])
    ax.set_ylim3d([-size, size])
    ax.set_zlim3d([-size, size])
    plt.savefig("JAX_orbits_3d.png", dpi=300)
    
def orbital_velocity(mass, radius):
    v = jnp.sqrt(mass/radius)
    return v


def quantity_test(orbit1, orbit2, M1, M2, h, T):
    traj1, traj2 = integrate_nbody(orbit1, orbit2, M1, M2, h, T)
    vx=(traj1[:,4]-traj2[:,4])
    vy=(traj1[:,5]-traj2[:,5])
    x_rel=(traj1[:,1]-traj2[:,1])
    y_rel=(traj1[:,2]-traj2[:,2])

    # m1m2=M1*M2
    E   = 0.5*(M1*(traj1[:,4]**2+traj1[:,5]**2) + M2*(traj2[:,4]**2+traj2[:,5]**2)) \
    + PointMassPotential(M1,M2).potential(traj1[:,1],traj1[:,2],traj1[:,3],traj2[:,1],traj2[:,2],traj2[:,3])

    dE  = (E-E[0])/E[0]

    L = x_rel*vy - y_rel*vx 
    dL  = (L-L[0])/L[0]
    return traj1, traj2, dE, dL

if __name__ == "__main__":
    jax.config.update('jax_enable_x64', True)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
    import numpy as np

    orbit1 = jnp.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.6, 0.0])
    orbit2 = jnp.array([0.0, -0.7, 0.0, 0.0, 0.0,-0.5, 0.0])
    orbit3 = jnp.array([0.0, 0, 0, -2, 0.5**0.5,-0.0, 0.0])
    orbit4 = jnp.array([0.0, 0, 2.8, 0.0, (1/2.8)**0.5,-0.0, 0.0])
    orbit5 = jnp.array([0.0, 0, 0, 1, 0.0,-1, 0.0])
    masses = jnp.array([0.5, 0.5, 0.001, 0.0003, 0.0002])  # (2,)

    # masses = jnp.array([1, 0.0001, 0.0003, 0.0008, 0.0002])  # (2,)
    M=masses[0]
    r1=3.8
    r2=1
    r3=1.9
    r4 =2.7
    # orbit1 = jnp.array([0.0, 0., 0.0, 0.0, 0.0, 0., 0.0])
    # orbit2 = jnp.array([0.0, r1, 0.0, 0.0, 0.0,orbital_velocity(M,r1), 0.0])
    # orbit3 = jnp.array([0.0, 0, r2, 0, orbital_velocity(M,r2) ,-0.0, 0.0])
    # orbit4 = jnp.array([0.0, 0, -r3, 0.0, -orbital_velocity(M,r3),-0.0, 0.0])
    # orbit5 = jnp.array([0.0, -r4, 0, 0, 0.0,-orbital_velocity(M,r4), 0.0])

    orbits = jnp.stack([orbit1, orbit2, orbit3, orbit4, orbit5])  # (2,7)

    h = 0.0008
    T = 50.0

    traj_com = integrate_nbody(orbits, masses, h, T)  # (num_steps, 2, 7)

    # Plotting (optional; similar to original)
    from matplotlib import pyplot as plt
    # plot 2-body result

    plot_2d = True 
    if plot_2d:
        plt.plot(traj_com[:, 0, 1], traj_com[:, 0, 2], markersize=.1, linewidth=1)
        plt.plot(traj_com[:, 1, 1], traj_com[:, 1, 2], markersize=.1, linewidth=1)
        plt.plot(traj_com[:, 2, 1], traj_com[:, 2, 2], markersize=.1, linewidth=1)
        plt.plot(traj_com[:, 3, 1], traj_com[:, 3, 2], markersize=.1, linewidth=1)
        plt.plot(traj_com[:, 4, 1], traj_com[:, 4, 2], markersize=.1, linewidth=1)

        plt.axis('equal')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.title("n-body (COM frame)")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig("JAX_orbits_nbody.png", dpi=300)
        plt.close()
    else:
        plot_3d_trajectories(traj_com, elev=35, azim=45)

    # plt.plot(traj_com[:,0],dE,label=r'$\Delta E/E_0$')
    # plt.plot(traj_com[:,0],dL,label=r'$\Delta L/L_0$')
    # plt.xlabel('time')
    # plt.ylabel('Relative Error')
    # plt.legend()
    # plt.savefig(save_path+"/JAX_orbits_errors.png", dpi=300)
