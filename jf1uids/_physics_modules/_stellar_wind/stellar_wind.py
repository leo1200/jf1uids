
import jax.numpy as jnp
import numpy as np
import jax
from astropy.io import fits
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.fluid import pressure_from_energy

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from functools import partial

from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids.units.unit_helpers import CodeUnits

from jf1uids._physics_modules._stellar_wind.stellar_wind_options import WindParams
from jf1uids._physics_modules._stellar_wind.stellar_wind_options import WindConfig, MEO, MEI, EI
from jf1uids._physics_modules._stellar_wind.stellar_wind_functions import get_current_wind_params, get_wind_parameters
from jf1uids._physics_modules._binary._binary_options import BinaryParams

from typing import Union

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _wind_injection(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams, 
    helper_data: HelperData, 
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
    binary_state: Union[None, Float[Array, "n"]] = None
) -> STATE_TYPE:
    """Inject stellar wind into the simulation.

    Args:
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    if config.dimensionality == 1:
        if config.wind_config.wind_injection_scheme == MEO:
            primitive_state = _wind_meo(params.wind_params, primitive_state, dt, helper_data, config.num_ghost_cells, config.wind_config.num_injection_cells, params.gamma)
        # elif config.wind_config.wind_injection_scheme == MEI:
        #     primitive_state = _wind_mei(params.wind_params, primitive_state, dt, helper_data, config.num_ghost_cells, config.wind_config.num_injection_cells, params.gamma, registered_variables)
        elif config.wind_config.wind_injection_scheme == EI:
            primitive_state = _wind_ei(params.wind_params, primitive_state, dt, helper_data, config.num_ghost_cells, config.wind_config.num_injection_cells, params.gamma, registered_variables)
        else:
            raise ValueError("Invalid wind injection scheme")
    elif config.dimensionality == 3:
        if config.wind_config.wind_injection_scheme == EI:
            # primitive_state = _wind_ei3D_radial(params.wind_params, primitive_state, dt, config, helper_data, config.num_ghost_cells, config.wind_config.num_injection_cells, params.gamma, registered_variables, current_time, binary_state)
            primitive_state = _wind_ei3D(params.wind_params, primitive_state, dt, config, helper_data, config.num_ghost_cells, config.wind_config.num_injection_cells, params.gamma, registered_variables, current_time, binary_state)
        else:
            raise ValueError("Invalid wind injection scheme")
    else:
        raise ValueError("Invalid dimensionality")

    return primitive_state

# ================= Wind injection schemes =================

# here we implement all the injection schemes from
# https://arxiv.org/abs/2107.14673

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells'])
def _wind_meo(wind_params: WindParams, primitive_state: Float[Array, "num_vars num_cells"], dt: Float[Array, ""], helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]]) -> Float[Array, "num_vars num_cells"]:
    """Inject stellar wind into the simulation by a mass-and-energy-overwrite scheme (MEO).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    # set density
    density_overwrite = wind_params.wind_mass_loss_rate / helper_data.cell_volumes[num_ghost_cells:num_injection_cells + num_ghost_cells] / wind_params.wind_final_velocity * (helper_data.outer_cell_boundaries[num_ghost_cells:num_injection_cells + num_ghost_cells] - helper_data.inner_cell_boundaries[num_ghost_cells:num_injection_cells + num_ghost_cells])
    primitive_state = primitive_state.at[0, num_ghost_cells:num_injection_cells + num_ghost_cells].set(density_overwrite)

    # set velocity
    primitive_state = primitive_state.at[1, num_ghost_cells:num_injection_cells + num_ghost_cells].set(wind_params.wind_final_velocity)

    # set pressure to the floor value
    primitive_state = primitive_state.at[2, num_ghost_cells:num_injection_cells + num_ghost_cells].set(wind_params.pressure_floor)

    return primitive_state

# @jaxtyped(typechecker=typechecker)
# @partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells', 'registered_variables'])
# def _wind_mei(wind_params: WindParams, primitive_state: Float[Array, "num_vars num_cells"], dt: Float[Array, ""], helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables) -> Float[Array, "num_vars num_cells"]:
#     """Inject stellar wind into the simulation by a momentum-and-energy-injection scheme (MEI).
    
#     Args:
#         wind_params: The wind parameters.
#         primitive_state: The primitive state array.
#         dt: The time step.
#         helper_data: The helper data.
#         num_ghost_cells: The number of ghost cells.
#         num_injection_cells: The number of injection cells.
#         gamma: The adiabatic index.
        
#     Returns:
#         The primitive state array with the stellar wind injected.
#     """

#     conservative_state = conserved_state_from_primitive(primitive_state, gamma, registered_variables)

#     V_inj = 4/3 * jnp.pi * helper_data.outer_cell_boundaries[num_injection_cells + num_ghost_cells]**3

#     drho = wind_params.wind_mass_loss_rate * dt / V_inj
#     dmomentum = wind_params.wind_final_velocity * drho
#     denergy = 0.5 * wind_params.wind_final_velocity**2 * drho

#     conservative_state = conservative_state.at[0, num_ghost_cells:num_injection_cells + num_ghost_cells].add(drho)
#     conservative_state = conservative_state.at[1, num_ghost_cells:num_injection_cells + num_ghost_cells].add(dmomentum)
#     conservative_state = conservative_state.at[2, num_ghost_cells:num_injection_cells + num_ghost_cells].add(denergy)

#     primitive_state = primitive_state_from_conserved(conservative_state, gamma, config, registered_variables)

#     return primitive_state

# not really ei
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells', 'registered_variables'])
def _wind_ei(wind_params: WindParams, primitive_state: Float[Array, "num_vars num_cells"], dt: Float[Array, ""], helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables) -> Float[Array, "num_vars num_cells"]:
    """Inject stellar wind into the simulation by an thermal-energy-injection scheme (EI).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    source_term = jnp.zeros_like(primitive_state)
    
    r = helper_data.volumetric_centers
    r_inj = r[num_injection_cells + 2]
    V = 4/3 * jnp.pi * r_inj**3

    # V = jnp.sum(helper_data.cell_volumes[num_ghost_cells:num_injection_cells + num_ghost_cells])

    # mass injection
    drho_dt = wind_params.wind_mass_loss_rate / V
    source_term = source_term.at[0, num_ghost_cells:num_injection_cells + num_ghost_cells].set(drho_dt)
    updated_density = primitive_state[0, num_ghost_cells:num_injection_cells + num_ghost_cells] + drho_dt * dt

    if registered_variables.wind_density_active:
        source_term = source_term.at[registered_variables.wind_density_index, num_ghost_cells:num_injection_cells + num_ghost_cells].set(drho_dt)

    # energy injection
    dE_dt = 0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate / V

    dp_dt = pressure_from_energy(dE_dt, updated_density, primitive_state[1, num_ghost_cells:num_injection_cells + num_ghost_cells], gamma)

    source_term = source_term.at[2, num_ghost_cells:num_injection_cells + num_ghost_cells].set(dp_dt)

    primitive_state = primitive_state + source_term * dt

    return primitive_state

# not really ei
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells', 'registered_variables'])
def dummy_multi_star_wind(wind_params: WindParams, primitive_state: STATE_TYPE, dt: Float[Array, ""], config: SimulationConfig, helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables) -> STATE_TYPE:
    
    star_positions = [jnp.array([0.2, 0.3, 0.5]), jnp.array([0.5, 0.7, 0.5]), jnp.array([0.7, 0.4, 0.5]), jnp.array([0.3, 0.6, 0.5])]

    for star_position in star_positions:

        r = jnp.linalg.norm(helper_data.geometric_centers - star_position, axis = -1)

        source_term = jnp.zeros_like(primitive_state)
        
        r_inj = num_injection_cells * config.grid_spacing
        V = 4/3 * jnp.pi * r_inj**3

        # for now only allow injection at the box center
        injection_mask = r <= r_inj - config.grid_spacing / 2

        # mass injection
        drho_dt = wind_params.wind_mass_loss_rate / V
        # source_term = source_term.at[registered_variables.density_index].set(jnp.where(injection_mask, drho_dt, source_term[registered_variables.density_index]))
        source_term = source_term.at[registered_variables.density_index].set(drho_dt * injection_mask)

        updated_density = primitive_state[registered_variables.density_index]
        updated_density = jnp.where(injection_mask > 0, updated_density + drho_dt * dt * injection_mask, updated_density)

        # energy injection
        dE_dt = 0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate / V
        u = jnp.sqrt(primitive_state[registered_variables.velocity_index.x]**2 + primitive_state[registered_variables.velocity_index.y]**2 + primitive_state[registered_variables.velocity_index.z]**2)
        dp_dt = pressure_from_energy(dE_dt, updated_density, u, gamma)
        
        source_term = source_term.at[registered_variables.pressure_index].set(dp_dt * injection_mask)

        primitive_state = primitive_state + source_term * dt

    return primitive_state


# # not really ei
# @jaxtyped(typechecker=typechecker)
# @partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells', 'registered_variables', 'config'])
# def _wind_ei3D(wind_params: WindParams, primitive_state: STATE_TYPE, dt: Float[Array, ""], config: SimulationConfig, helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables) -> STATE_TYPE:
#     """Inject stellar wind into the simulation by an thermal-energy-injection scheme (EI).

#     Args:
#         wind_params: The wind parameters.
#         primitive_state: The primitive state array.
#         dt: The time step.
#         helper_data: The helper data.
#         num_ghost_cells: The number of ghost cells.
#         num_injection_cells: The number of injection cells.
#         gamma: The adiabatic index.

#     Returns:
#         The primitive state array with the stellar wind injected.
#     """

#     source_term = jnp.zeros_like(primitive_state)
    
#     r_inj = num_injection_cells * config.grid_spacing
#     V = 4/3 * jnp.pi * r_inj**3

#     # for now only allow injection at the box center
#     injection_mask = helper_data.r <= r_inj - config.grid_spacing / 2
#     # overlap_weights = (r_inj + config.grid_spacing / 2 - helper_data.r) / config.grid_spacing
#     # overlap_mask = (helper_data.r > r_inj - config.grid_spacing / 2) & (helper_data.r < r_inj + config.grid_spacing / 2)
#     # overlap_weights = overlap_weights * overlap_mask
#     # injection_mask = injection_mask | overlap_mask
#     # injection_mask = injection_mask / jnp.sum(injection_mask * config.grid_spacing**3) * V

#     # mass injection
#     drho_dt = wind_params.wind_mass_loss_rate / V
#     # source_term = source_term.at[registered_variables.density_index].set(jnp.where(injection_mask, drho_dt, source_term[registered_variables.density_index]))
#     source_term = source_term.at[registered_variables.density_index].set(drho_dt * injection_mask)

#     updated_density = primitive_state[registered_variables.density_index]
#     updated_density = jnp.where(injection_mask > 0, updated_density + drho_dt * dt * injection_mask, updated_density)

#     # scale down the velocity in the primitive state to conserve momentum
#     # density_ratio = updated_density / primitive_state[registered_variables.density_index]
#     # primitive_state = primitive_state.at[registered_variables.velocity_index.x].set(jnp.where(injection_mask, primitive_state[registered_variables.velocity_index.x] * density_ratio, primitive_state[registered_variables.velocity_index.x]))
#     # primitive_state = primitive_state.at[registered_variables.velocity_index.y].set(jnp.where(injection_mask, primitive_state[registered_variables.velocity_index.y] * density_ratio, primitive_state[registered_variables.velocity_index.y]))
#     # primitive_state = primitive_state.at[registered_variables.velocity_index.z].set(jnp.where(injection_mask, primitive_state[registered_variables.velocity_index.z] * density_ratio, primitive_state[registered_variables.velocity_index.z]))

#     # energy injection
#     dE_dt = 0.5 * wind_params.wind_final_velocity**2 * wind_params.wind_mass_loss_rate / V
#     u = jnp.sqrt(primitive_state[registered_variables.velocity_index.x]**2 + primitive_state[registered_variables.velocity_index.y]**2 + primitive_state[registered_variables.velocity_index.z]**2)
#     dp_dt = pressure_from_energy(dE_dt, updated_density, u, gamma)
    
#     source_term = source_term.at[registered_variables.pressure_index].set(dp_dt * injection_mask)

#     primitive_state = primitive_state + source_term * dt

#     return primitive_state
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells', 'registered_variables', 'config'])
def _wind_ei3D_radial(
    wind_params: WindParams,
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    num_ghost_cells: int,
    num_injection_cells: int,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
    binary_state: Union[None, Float[Array, "n"]] = None
) -> STATE_TYPE:
    """
        Inject stellar wind energy/mass from multiple sources.
        NOTE: changed to triangular-shaped cloud injection with quadratic
              weights (weight = (1 - r/r_inj)**2 for r <= r_inj).
    """

    source_term = jnp.zeros_like(primitive_state)
    # If you want per-source injection radii, pass an array instead and broadcast accordingly
    r_inj = num_injection_cells * config.grid_spacing
    V = 4.0 / 3.0 * jnp.pi * r_inj ** 3

    y = helper_data.geometric_centers[..., 0] - config.box_size / 2   #x and y axis are exchanged because meshgrid indexing=xy
    x = helper_data.geometric_centers[..., 1] - config.box_size / 2
    z = helper_data.geometric_centers[..., 2] - config.box_size / 2

    if config.binary_config.binary == True:
        state = binary_state.reshape(-1, 7)
        source_positions = state[:, 1:4]
    else:
        source_positions = wind_params.wind_injection_positions
    dx = x[None, ...] - source_positions[:, 0, None, None, None]
    dy = y[None, ...] - source_positions[:, 1, None, None, None]
    dz = z[None, ...] - source_positions[:, 2, None, None, None]
    dist = jnp.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # >>> CHANGED: compute a smooth triangular-shaped (quadratic) radial weight:
    # weight(r) = (1 - r/r_inj)^2 for r <= r_inj, else 0
    # This replaces the old sharp mask that used (dist <= r_inj - dx/2).
    # The kernel is radial and will be normalized per source so that integrated
    # injected mass = mass_rate for that source.
    # raw_weight = jnp.where(dist <= r_inj, (1.0 - dist / r_inj) ** 2, 0.0)  # >>> CHANGED: raw per-cell weight (N, Nx, Ny, Nz)
    r_inj = num_injection_cells * config.grid_spacing
    cell_volume = config.grid_spacing ** 3  
    
    # radial Gaussian kernel (truncated at r_inj)
    sigma = r_inj / 2.0                 # tune: smaller sigma = sharper shell
    raw_weight = jnp.where(dist <= r_inj,
                       jnp.exp(-(dist/sigma)**2),
                       0.0)
    # >>> CHANGED: Normalize weights per source so the sum(weight * dV) == 1.0
    # sum_weights has shape (N,)
    sum_weights = jnp.sum(raw_weight, axis=(1, 2, 3))
    # avoid division by zero for sources with no overlapping cells (rare) by
    # replacing zero with 1.0 (raw_weight will be zero so result is safe)
    sum_weights_safe = jnp.where(sum_weights > 0.0, sum_weights, 1.0)
    normalization = sum_weights_safe * cell_volume  # >>> CHANGED: S = sum(f_i) * dV

    # normalized weight per cell (dimensionless); sums to 1 when multiplied by cell_volume
    weight = raw_weight / normalization[:, None, None, None]  # >>> CHANGED
    
    ##### supersampling #####################
    # 8-point subcell offsets in units of grid_spacing
    offsets = jnp.array([[0.25,0.25,0.25],[0.75,0.25,0.25],[0.25,0.75,0.25],[0.75,0.75,0.25],
                     [0.25,0.25,0.75],[0.75,0.25,0.75],[0.25,0.75,0.75],[0.75,0.75,0.75]])

    def radial_kernel(dist, r_inj):
        sigma = r_inj * 0.4           # tune
        return jnp.where(dist <= r_inj, jnp.exp(-(dist/sigma)**2), 0.0)

    # accumulate subsamples
    raw_weight = 0.0
    for off in offsets:
        dxs = (x[None,...] + off[0]*config.gri2d_spacing) - source_positions[:,0,None,None,None]
        dys = (y[None,...] + off[1]*config.grid_spacing) - source_positions[:,1,None,None,None]
        dzs = (z[None,...] + off[2]*config.grid_spacing) - source_positions[:,2,None,None,None]
        dist_sub = jnp.sqrt(dxs**2 + dys**2 + dzs**2)
        raw_weight = raw_weight + radial_kernel(dist_sub, r_inj)

    raw_weight = raw_weight / offsets.shape[0]   # average subsamples

    # normalize per source as before
    sum_weights = jnp.sum(raw_weight, axis=(1,2,3))
    sum_weights_safe = jnp.where(sum_weights > 0.0, sum_weights, 1.0)
    normalization = sum_weights_safe * cell_volume
    weight = raw_weight / normalization[:,None,None,None]

    if config.wind_config.real_wind_params:
        time_value, mass_rates_value, vel_scales_value = wind_params.real_params
        mass_rates, vel_scales = get_current_wind_params(mass_rates_value, vel_scales_value, current_time, time_value)
    else:
        mass_rates = wind_params.wind_mass_loss_rates  
        vel_scales = wind_params.wind_final_velocities 

    # >>> CHANGED: distribute mass rate over cells using normalized weight.
    # drho_dt_sources has shape (N, Nx, Ny, Nz); sum over sources
    # each source contributes mass_rate * weight_cell (units: mass/time/volume)
    drho_dt_sources = mass_rates[:, None, None, None] * weight  # >>> CHANGED
    drho_dt = jnp.sum(drho_dt_sources, axis=0)  

    # >>> CHANGED: energy injection rate per source follows the same weighting
    dE_dt_sources = (0.5 * (vel_scales ** 2) * mass_rates)[:, None, None, None] * weight  # >>> CHANGED
    dE_dt = jnp.sum(dE_dt_sources, axis=0) 

    source_term = source_term.at[registered_variables.density_index].set(drho_dt)

    # Update density in primitives
    updated_density = primitive_state[registered_variables.density_index]
    updated_density = jnp.where(drho_dt > 0.0, updated_density + drho_dt * dt, updated_density)

    # supply the multi-source dE_dt and updated_density.
    u = jnp.sqrt(
        primitive_state[registered_variables.velocity_index.x] ** 2
        + primitive_state[registered_variables.velocity_index.y] ** 2
        + primitive_state[registered_variables.velocity_index.z] ** 2
    )

    dp_dt = pressure_from_energy(dE_dt, updated_density, u, gamma)
    source_term = source_term.at[registered_variables.pressure_index].set(dp_dt)
    primitive_state = primitive_state + source_term * dt

    return primitive_state


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells', 'registered_variables', 'config'])
def _wind_ei3D(
    wind_params: WindParams,
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    num_ghost_cells: int,
    num_injection_cells: int,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
    binary_state: Union[None, Float[Array, "n"]] = None
) -> STATE_TYPE:
    """Inject stellar wind energy/mass from multiple sources.
     * Added arguments: `source_positions`, `source_mass_loss_rates`,
       `source_final_velocities` so the function receives N independent
       wind sources.
    """

    source_term = jnp.zeros_like(primitive_state)
    # If you want per-source injection radii, pass an array instead and broadcast accordingly
    r_inj = num_injection_cells * config.grid_spacing
    V = 4.0 / 3.0 * jnp.pi * r_inj ** 3

    y = helper_data.volumetric_centers[..., 0] - config.box_size / 2   #x and y axis are exchanged because meshgrid indexing=xy
    x = helper_data.volumetric_centers[..., 1] - config.box_size / 2
    z = helper_data.volumetric_centers[..., 2] - config.box_size / 2

    if config.binary_config.binary == True:
        state = binary_state.reshape(-1, 7)
        source_positions = state[:, 1:4]
    else:
        source_positions = wind_params.wind_injection_positions
    dx = x[None, ...] - source_positions[:, 0, None, None, None]
    dy = y[None, ...] - source_positions[:, 1, None, None, None]
    dz = z[None, ...] - source_positions[:, 2, None, None, None]
    dist = jnp.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # include cells whose cell center is within (r_inj - dx/2) from the source center.
    per_source_mask = (dist <= (r_inj - config.grid_spacing / 2.0)).astype(primitive_state.dtype)
    
    if config.wind_config.real_wind_params:
        time_value, mass_rates_value, vel_scales_value = wind_params.real_params
        mass_rates, vel_scales = get_current_wind_params(mass_rates_value, vel_scales_value, current_time, time_value)
    else:
        mass_rates = wind_params.wind_mass_loss_rates  
        vel_scales = wind_params.wind_final_velocities 

    # jax.debug.print("mass_rates: {m}", m=mass_rates)
    # jax.debug.print("vel_scales: {v}", v=vel_scales)
    # drho_dt_sources has shape (N, Nx, Ny, Nz); sum over sources   
    drho_dt_sources = (mass_rates[:, None, None, None] / V) * per_source_mask
    drho_dt = jnp.sum(drho_dt_sources, axis=0)  

    # energy injection rate per source 
    dE_dt_sources = (0.5 * (vel_scales ** 2) * mass_rates)[:, None, None, None] / V * per_source_mask
    dE_dt = jnp.sum(dE_dt_sources, axis=0) 

    # multi-source summed `drho_dt`.
    source_term = source_term.at[registered_variables.density_index].set(drho_dt)

    # Update density in primitives
    updated_density = primitive_state[registered_variables.density_index]
    updated_density = jnp.where(drho_dt > 0.0, updated_density + drho_dt * dt, updated_density)

    # supply the multi-source dE_dt and updated_density.
    u = jnp.sqrt(
        primitive_state[registered_variables.velocity_index.x] ** 2
        + primitive_state[registered_variables.velocity_index.y] ** 2
        + primitive_state[registered_variables.velocity_index.z] ** 2
    )

    dp_dt = pressure_from_energy(dE_dt, updated_density, u, gamma)
    source_term = source_term.at[registered_variables.pressure_index].set(dp_dt)
    primitive_state = primitive_state + source_term * dt

    return primitive_state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells', 'registered_variables', 'config'])
def _wind_ei3D_tsc(
    wind_params: WindParams,
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    num_ghost_cells: int,
    num_injection_cells: int,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
    binary_state: Union[None, Float[Array, "n"]] = None
) -> STATE_TYPE:
    """
    Inject stellar wind energy/mass from multiple sources using a TRUE
    separable 3D TSC (Triangular-Shaped-Cloud) kernel.

    Notes on changes:
    - Replaces the isotropic radial quadratic kernel with a separable 1D TSC
      kernel along each axis: w3D = w(x) * w(y) * w(z).
    - The per-axis TSC uses the standard piecewise formula:
        s = |dx| / (grid_spacing * num_injection_cells_scale)
        if 0 <= s < 0.5:   w = 3/4 - s^2
        if 0.5 <= s < 1.5: w = 0.5 * (1.5 - s)^2
        else:              w = 0
      with support up to s < 1.5 (so physical support radius = 1.5 * scale).
    - For compatibility with the user's previous interface, the TSC scale is
      taken to be `num_injection_cells * grid_spacing` so that
      num_injection_cells=1 corresponds to a kernel spanning ≈3 cells (±1.5 dx).
    - We normalize per source so sum(weight * dV) == 1, ensuring integrated
      injected mass == mass_rate for each source.
    """

    source_term = jnp.zeros_like(primitive_state)

    # cell volume (used for normalization). >>> CHANGED TO TSC uses same normalization approach.
    cell_volume = config.grid_spacing ** 3

    # grid-centered coordinates (same as before)
    y = helper_data.geometric_centers[..., 0] - config.box_size / 2   # x/y swapped due to meshgrid indexing=xy
    x = helper_data.geometric_centers[..., 1] - config.box_size / 2
    z = helper_data.geometric_centers[..., 2] - config.box_size / 2

    # get source positions (binary handling unchanged)
    if config.binary_config.binary == True:
        state = binary_state.reshape(-1, 7)
        source_positions = state[:, 1:4]
    else:
        source_positions = wind_params.wind_injection_positions

    # dx, dy, dz shape: (N_sources, Nx, Ny, Nz)
    dx = x[None, ...] - source_positions[:, 0, None, None, None]
    dy = y[None, ...] - source_positions[:, 1, None, None, None]
    dz = z[None, ...] - source_positions[:, 2, None, None, None]

    # >>> CHANGED TO TSC:
    # Define per-axis TSC kernel (separable). We scale distances by:
    # scale = grid_spacing * num_injection_cells. If num_injection_cells == 1,
    # this reduces to the canonical TSC support of 1.5 * dx.
    # s = abs(delta) / scale
    # support: s < 1.5
    scale = config.grid_spacing * jnp.maximum(1, num_injection_cells)  # avoid 0
    sx = jnp.abs(dx) / scale
    sy = jnp.abs(dy) / scale
    sz = jnp.abs(dz) / scale

    # 1D TSC function (vectorized via jnp.where)
    # w_1d(s) = { 3/4 - s^2              , 0 <= s < 0.5
    #            { 0.5*(1.5 - s)^2        , 0.5 <= s < 1.5
    #            { 0                      , s >= 1.5
    def tsc_1d(s):
        w = jnp.where(
            s < 0.5,
            0.75 - s ** 2,
            jnp.where(s < 1.5, 0.5 * (1.5 - s) ** 2, 0.0),
        )
        return w

    wx = tsc_1d(sx)
    wy = tsc_1d(sy)
    wz = tsc_1d(sz)

    # separable 3D weight (dimensionless)
    # raw_weight shape: (N_sources, Nx, Ny, Nz)
    raw_weight = wx * wy * wz  # >>> CHANGED TO TSC: separable product kernel
    
    # >>> CHANGED TO TSC: Normalize per-source so sum(raw_weight * cell_volume) == 1.
    # sum_weights (N_sources,)
    sum_weights = jnp.sum(raw_weight, axis=(1, 2, 3))
    # avoid division by zero for sources with no overlapping cells
    sum_weights_safe = jnp.where(sum_weights > 0.0, sum_weights, 1.0)
    normalization = sum_weights_safe * cell_volume  # has units of volume
    weight = raw_weight / normalization[:, None, None, None]  # now has units 1/volume

    # fetch (possibly time-dependent) mass rates and velocity scales (unchanged)
    if config.wind_config.real_wind_params:
        time_value, mass_rates_value, vel_scales_value = wind_params.real_params
        mass_rates, vel_scales = get_current_wind_params(mass_rates_value, vel_scales_value, current_time, time_value)
    else:
        mass_rates = wind_params.wind_mass_loss_rates
        vel_scales = wind_params.wind_final_velocities

    # distribute mass rate over cells using the normalized TSC weights
    # drho_dt_sources shape: (N_sources, Nx, Ny, Nz)
    drho_dt_sources = mass_rates[:, None, None, None] * weight  # mass/time/volume per cell from each source
    drho_dt = jnp.sum(drho_dt_sources, axis=0)  # total dm/(dt dV) in each cell

    # energy injection (per-source kinetic energy -> same separable weighting)
    dE_dt_sources = (0.5 * (vel_scales ** 2) * mass_rates)[:, None, None, None] * weight
    dE_dt = jnp.sum(dE_dt_sources, axis=0)

    # put mass source into source_term and update primitives (same as before)
    source_term = source_term.at[registered_variables.density_index].set(drho_dt)

    # Update density in primitives (unchanged logic)
    updated_density = primitive_state[registered_variables.density_index]
    updated_density = jnp.where(drho_dt > 0.0, updated_density + drho_dt * dt, updated_density)

    # velocity magnitude (unchanged)
    u = jnp.sqrt(
        primitive_state[registered_variables.velocity_index.x] ** 2
        + primitive_state[registered_variables.velocity_index.y] ** 2
        + primitive_state[registered_variables.velocity_index.z] ** 2
    )

    # convert energy injection to pressure change (same helper)
    dp_dt = pressure_from_energy(dE_dt, updated_density, u, gamma)
    source_term = source_term.at[registered_variables.pressure_index].set(dp_dt)

    # finally update primitives
    primitive_state = primitive_state + source_term * dt

    return primitive_state


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_ghost_cells', 'num_injection_cells', 'registered_variables'])
def _wind_ei3D_superres(wind_params: WindParams, primitive_state: STATE_TYPE, dt: Float[Array, ""], config: SimulationConfig, helper_data: HelperData, num_ghost_cells: int, num_injection_cells: int, gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables) -> STATE_TYPE:
    """Inject stellar wind into the simulation by an thermal-energy-injection scheme (EI).

    Args:
        wind_params: The wind parameters.
        primitive_state: The primitive state array.
        dt: The time step.
        helper_data: The helper data.
        num_ghost_cells: The number of ghost cells.
        num_injection_cells: The number of injection cells.
        gamma: The adiabatic index.

    Returns:
        The primitive state array with the stellar wind injected.
    """

    source_term = jnp.zeros_like(primitive_state)
    r_inj = num_injection_cells * config.grid_spacing

    total_mass_change = wind_params.wind_mass_loss_rate * dt
    total_energy_change = 0.5 * wind_params.wind_final_velocity**2 * total_mass_change

    superres_factor = 8
    superres_grid_size = superres_factor * num_injection_cells * 2
    superres_grid_spacing = config.grid_spacing / superres_factor
    
    half_width = superres_grid_size * superres_grid_spacing / 2

    x = jnp.linspace(-half_width, half_width, superres_grid_size)
    y = jnp.linspace(-half_width, half_width, superres_grid_size)
    z = jnp.linspace(-half_width, half_width, superres_grid_size)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    R = jnp.sqrt(X**2 + Y**2 + Z**2)
    superres_injection_weights = R <= r_inj
    superres_injection_weights = superres_injection_weights / jnp.sum(superres_injection_weights)


    # sum pool down the mask to get to a mask of size (num_injection_cells * 2)^3
    superres_injection_weights = superres_injection_weights.reshape((num_injection_cells * 2, superres_factor,
                                   num_injection_cells * 2, superres_factor,
                                   num_injection_cells * 2, superres_factor)).sum(axis=(1, 3, 5))
    

    injection_weights = jnp.zeros_like(primitive_state[0])
    half_index = primitive_state[0].shape[0] // 2
    injection_weights = injection_weights.at[half_index - num_injection_cells:half_index + num_injection_cells, half_index - num_injection_cells:half_index + num_injection_cells, half_index - num_injection_cells:half_index + num_injection_cells].set(superres_injection_weights)

    source_term = source_term.at[registered_variables.density_index].set(total_mass_change * injection_weights / (config.grid_spacing**3))
    gamma = 4/3
    source_term = source_term.at[registered_variables.pressure_index].set(total_energy_change * (gamma - 1) * injection_weights / (config.grid_spacing**3))

    primitive_state = primitive_state + source_term

    return primitive_state

