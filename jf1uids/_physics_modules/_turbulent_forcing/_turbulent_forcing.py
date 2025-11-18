import jax
import jax.numpy as jnp
from functools import partial

from jf1uids._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingParams
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.variable_registry.registered_variables import RegisteredVariables

@partial(jax.jit, static_argnames=["config"])
def _create_forcing_field(
    key, 
    config: SimulationConfig,
):
    
    # in our case we assume a cubic box
    # with equal number of cells in each direction

    xsize = config.box_size
    ysize = config.box_size
    zsize = config.box_size

    nx = config.num_cells
    ny = config.num_cells
    nz = config.num_cells
    
    # Define wavenumbers using fftfreq (more efficient than manual loops)
    kx = jnp.fft.fftfreq(nx, d=xsize/(2*jnp.pi*nx)) * (2*jnp.pi)
    ky = jnp.fft.fftfreq(ny, d=ysize/(2*jnp.pi*ny)) * (2*jnp.pi)
    kz = jnp.fft.fftfreq(nz, d=zsize/(2*jnp.pi*nz)) * (2*jnp.pi)
    
    # Use broadcasting instead of meshgrid (memory efficient)
    kx_3d = kx.reshape(nx, 1, 1)
    ky_3d = ky.reshape(1, ny, 1)
    kz_3d = kz.reshape(1, 1, nz)
    
    k_squared = kx_3d**2 + ky_3d**2 + kz_3d**2
    kk = jnp.sqrt(k_squared)
    
    # Power spectrum parameters
    kpk = 2.0 * (2.0 * jnp.pi / xsize)
    Pk = kk**6 * jnp.exp(-8.0 * kk / kpk)
    
    key, subkey = jax.random.split(key)
    ran_nums = jax.random.uniform(subkey, shape=(6, nx, ny, nz))
    
    # Box-Muller transform for Gaussian deviates
    sqrt_Pk = jnp.sqrt(Pk)
    temp1 = sqrt_Pk * jnp.sqrt(-2.0 * jnp.log(ran_nums[0] + 1e-10)) * jnp.cos(2.0 * jnp.pi * ran_nums[1])
    temp2 = sqrt_Pk * jnp.sqrt(-2.0 * jnp.log(ran_nums[0] + 1e-10)) * jnp.sin(2.0 * jnp.pi * ran_nums[1])
    
    temp3 = sqrt_Pk * jnp.sqrt(-2.0 * jnp.log(ran_nums[2] + 1e-10)) * jnp.cos(2.0 * jnp.pi * ran_nums[3])
    temp4 = sqrt_Pk * jnp.sqrt(-2.0 * jnp.log(ran_nums[2] + 1e-10)) * jnp.sin(2.0 * jnp.pi * ran_nums[3])
    
    temp5 = sqrt_Pk * jnp.sqrt(-2.0 * jnp.log(ran_nums[4] + 1e-10)) * jnp.cos(2.0 * jnp.pi * ran_nums[5])
    temp6 = sqrt_Pk * jnp.sqrt(-2.0 * jnp.log(ran_nums[4] + 1e-10)) * jnp.sin(2.0 * jnp.pi * ran_nums[5])
    
    # Create complex forcing fields
    cwx = temp1 + 1j * temp2
    cwy = temp3 + 1j * temp4
    cwz = temp5 + 1j * temp6
    
    # Set DC mode to zero
    cwx = cwx.at[0, 0, 0].set(0.0 + 0.0j)
    cwy = cwy.at[0, 0, 0].set(0.0 + 0.0j)
    cwz = cwz.at[0, 0, 0].set(0.0 + 0.0j)
    
    # Make divergence-free (project out compressible component)
    # Avoid division by zero with safe k_squared
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    
    # Compute divergence in Fourier space
    div_k = (kx_3d * cwx + ky_3d * cwy + kz_3d * cwz) / k_squared_safe
    
    # Set DC component of divergence to zero
    div_k = div_k.at[0, 0, 0].set(0.0 + 0.0j)
    
    # Remove divergent component
    cwx = cwx - kx_3d * div_k
    cwy = cwy - ky_3d * div_k
    cwz = cwz - kz_3d * div_k

    wx_real = jnp.real(jnp.fft.ifftn(cwx))
    wy_real = jnp.real(jnp.fft.ifftn(cwy))
    wz_real = jnp.real(jnp.fft.ifftn(cwz))
    
    return key, wx_real, wy_real, wz_real

@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _apply_forcing(
    key,
    primitive_state,
    dt,
    turbulent_forcing_params: TurbulentForcingParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    
    key, wx_real, wy_real, wz_real = _create_forcing_field(key, config)

    Edot = turbulent_forcing_params.energy_injection_rate
    dtforc = dt
    dV = config.grid_spacing**3
    
    # get density and velocities
    rho = primitive_state[registered_variables.density_index]
    u = primitive_state[registered_variables.velocity_index.x]
    v = primitive_state[registered_variables.velocity_index.y]
    w = primitive_state[registered_variables.velocity_index.z]

    # compute terms for quadratic equation
    # a * amp^2 + b * amp + c = 0
    tempa = 0.5 * jnp.sum(rho * (wx_real**2 + wy_real**2 + wz_real**2))
    tempb = jnp.sum(rho * u * wx_real + rho * v * wy_real + rho * w * wz_real)
    tempc = -Edot * dtforc / dV
    
    # solve quadratic equation for amplitude
    discriminant = tempb**2 - 4.0 * tempa * tempc
    
    # conditional to handle edge cases
    amp = jax.lax.cond(
        (discriminant >= 0) & (jnp.abs(tempa) > 1e-10),
        lambda: (-tempb + jnp.sqrt(discriminant)) / (2.0 * tempa),
        lambda: 0.0
    )
    
    # apply forcing to momentum
    primitive_state = primitive_state.at[registered_variables.velocity_index.x].add(
        amp * wx_real
    )
    primitive_state = primitive_state.at[registered_variables.velocity_index.y].add(
        amp * wy_real
    )
    primitive_state = primitive_state.at[registered_variables.velocity_index.z].add(
        amp * wz_real
    )
    return key, primitive_state


# @partial(jax.jit, static_argnames=["config", "registered_variables"])
# def apply_forcing(
#     key,
#     conserved_state,
#     dt,
#     turbulent_forcing_params: TurbulentForcingParams,
#     config: SimulationConfig,
#     registered_variables: RegisteredVariables,
# ):
    
#     key, wx_real, wy_real, wz_real = create_forcing_field(key, config)

#     Edot = turbulent_forcing_params.energy_injection_rate
#     dtforc = dt
#     dV = config.grid_spacing**3
    
#     # get density and velocities
#     rho = conserved_state[registered_variables.density_index]
#     rhou = conserved_state[registered_variables.momentum_index.x]
#     rhov = conserved_state[registered_variables.momentum_index.y]
#     rhow = conserved_state[registered_variables.momentum_index.z]

#     # compute terms for quadratic equation
#     # a * amp^2 + b * amp + c = 0
#     tempa = 0.5 * jnp.sum(rho * (wx_real**2 + wy_real**2 + wz_real**2))
#     tempb = jnp.sum(rhou * wx_real + rhov * wy_real + rhow * wz_real)
#     tempc = -Edot * dtforc / dV
    
#     # solve quadratic equation for amplitude
#     discriminant = tempb**2 - 4.0 * tempa * tempc
    
#     # conditional to handle edge cases
#     amp = jax.lax.cond(
#         (discriminant >= 0) & (jnp.abs(tempa) > 1e-10),
#         lambda: (-tempb + jnp.sqrt(discriminant)) / (2.0 * tempa),
#         lambda: 0.0
#     )
    
#     # apply forcing to momentum
#     conserved_state = conserved_state.at[registered_variables.momentum_index.x].add(
#         rho * amp * wx_real
#     )
#     conserved_state = conserved_state.at[registered_variables.momentum_index.y].add(
#         rho * amp * wy_real
#     )
#     conserved_state = conserved_state.at[registered_variables.momentum_index.z].add(
#         rho * amp * wz_real
#     )

#     # update the total energy
#     # kinetic energy = 0.5 * rho * u_squared
#     # where rho u -> rho (u + amp w)
#     # -> change in kinetic energy = 0.5 * rho * [ (u + amp w)^2 - u^2 ]
#     new_squared_velocity = (
#         (rhou / rho + amp * wx_real) ** 2
#         + (rhov / rho + amp * wy_real) ** 2
#         + (rhow / rho + amp * wz_real) ** 2
#     )
#     old_squared_velocity = (
#         (rhou / rho) ** 2
#         + (rhov / rho) ** 2
#         + (rhow / rho) ** 2
#     )
#     delta_kinetic_energy = 0.5 * rho * (new_squared_velocity - old_squared_velocity)
#     conserved_state = conserved_state.at[registered_variables.energy_index].add(delta_kinetic_energy)
    
#     return key, conserved_state