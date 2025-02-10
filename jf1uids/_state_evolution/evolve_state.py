# general imports
import jax.numpy as jnp
import jax
from functools import partial

# runtime debugging
from jax.experimental import checkify

# type checking imports
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# general jf1uids imports
from jf1uids._physics_modules._mhd.mhd_maths import magnetic_update
from jf1uids._physics_modules._self_gravity._self_gravity import _apply_self_gravity
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import CARTESIAN, HLL, HLLC, SPHERICAL, STATE_TYPE, SimulationConfig

from jf1uids._geometry.geometric_terms import _pressure_nozzling_source
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import gas_pressure_from_primitives_with_crs
from jf1uids._state_evolution.reconstruction import _reconstruct_at_interface
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids.fluid_equations.fluid import primitive_state_from_conserved, conserved_state_from_primitive
from jf1uids._riemann_solver.hll import _hll_solver, _hllc_solver

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables', 'axis'])
def _evolve_state_along_axis(
    primitive_state: STATE_TYPE,
    dx: Union[float, Float[Array, ""]],
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    axis: int
) -> STATE_TYPE:
    
    primitive_state = _boundary_handler(primitive_state, config)

    # get conserved variables
    conservative_states = conserved_state_from_primitive(primitive_state, gamma, config, registered_variables)

    num_cells = primitive_state.shape[axis]

    # flux in x-direction
    if config.first_order_fallback:
        primitive_state_left = jax.lax.slice_in_dim(primitive_state, 1, num_cells - 2, axis = axis)
        primitive_state_right = jax.lax.slice_in_dim(primitive_state, 2, num_cells - 1, axis = axis)
    else:
        primitive_state_left, primitive_state_right = _reconstruct_at_interface(primitive_state, dt, gamma, config, helper_data, registered_variables, axis)
    
    if config.riemann_solver == HLL:
        fluxes = _hll_solver(primitive_state_left, primitive_state_right, gamma, config, registered_variables, axis)
    elif config.riemann_solver == HLLC:
        fluxes = _hllc_solver(primitive_state_left, primitive_state_right, gamma, config, registered_variables, axis)
    else:
        raise ValueError("Riemann solver not supported.")

    flux_length = fluxes.shape[axis]

    # ================ update the conserved variables =================

    # usual cartesian case
    if config.geometry == CARTESIAN:
        conserved_change = -1 / dx * (jax.lax.slice_in_dim(fluxes, 1, flux_length, axis = axis) - jax.lax.slice_in_dim(fluxes, 0, flux_length - 1, axis = axis)) * dt

    # in spherical geometry, we have to take special care
    elif config.geometry == SPHERICAL and config.dimensionality == 1 and axis == 1:
        r = helper_data.geometric_centers
        r_hat_alpha = helper_data.r_hat_alpha

        alpha = config.geometry

        r_plus_half = r.at[config.num_ghost_cells:-config.num_ghost_cells].add(dx / 2)
        r_minus_half = r.at[config.num_ghost_cells:-config.num_ghost_cells].add(-dx / 2)

        # calculate the source terms
        nozzling_source = _pressure_nozzling_source(primitive_state, config, helper_data, registered_variables)

        # update the conserved variables using the fluxes and source terms
        conserved_change = 1 / r_hat_alpha[config.num_ghost_cells:-config.num_ghost_cells] * (
            - (r_plus_half[config.num_ghost_cells:-config.num_ghost_cells] ** alpha * fluxes[:, 1:] - r_minus_half[config.num_ghost_cells:-config.num_ghost_cells] ** alpha * fluxes[:, :-1]) / dx
            + nozzling_source[:, 1:-1]
        ) * dt

    # misconfiguration
    else:
        raise ValueError("Geometry and dimensionality combination not supported.")
    
    # =================================================================

    conservative_states = conservative_states.at[tuple(slice(config.num_ghost_cells, -config.num_ghost_cells) if i == axis else slice(None) for i in range(conservative_states.ndim))].add(conserved_change)

    primitive_state = primitive_state_from_conserved(conservative_states, gamma, config, registered_variables)
    primitive_state = _boundary_handler(primitive_state, config)

    # check if the pressure is still positive
    p = primitive_state[registered_variables.pressure_index]
    rho = primitive_state[registered_variables.density_index]

    if config.runtime_debugging:
        checkify.check(jnp.all(p >= 0), "pressure needs to be non-negative, minimum pressure {pmin} at index {index}", pmin=jnp.min(p), index=jnp.unravel_index(jnp.argmin(p), p.shape))
        checkify.check(jnp.all(rho >= 0), "density needs to be non-negative, minimum density {rhomin} at index {index}", rhomin=jnp.min(rho), index=jnp.unravel_index(jnp.argmin(rho), rho.shape))
    
    return primitive_state


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _evolve_gas_state(
    primitive_state: STATE_TYPE,
    dx: Union[float, Float[Array, ""]], dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig, helper_data: HelperData,
    registered_variables: RegisteredVariables
) -> STATE_TYPE:
    """Evolve the primitive state array.

    Args:
        primitive_state: The primitive state array.
        dx: The cell width.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.

    Returns:
        The evolved primitive state array.
    """
    if config.dimensionality == 1:
        primitive_state = _evolve_state_along_axis(primitive_state, dx, dt, gamma, config, helper_data, registered_variables, 1)

        if config.self_gravity:
            primitive_state = _apply_self_gravity(primitive_state, config, registered_variables, gamma, dt)

    elif config.dimensionality == 2:

        if config.self_gravity:
            primitive_state = _apply_self_gravity(primitive_state, config, registered_variables, gamma, dt)

        primitive_state = _evolve_state_along_axis(primitive_state, dx, dt/2, gamma, config, helper_data, registered_variables, 1)
        primitive_state = _evolve_state_along_axis(primitive_state, dx, dt, gamma, config, helper_data, registered_variables, 2)



        primitive_state = _evolve_state_along_axis(primitive_state, dx, dt/2, gamma, config, helper_data, registered_variables, 1)
    elif config.dimensionality == 3:

        if config.self_gravity:
            primitive_state = _apply_self_gravity(primitive_state, config, registered_variables, gamma, dt)


        # advance in x by dt/2 -> y by dt/2 -> z by dt -> y by dt/2 -> x by dt/2
        primitive_state = _evolve_state_along_axis(primitive_state, dx, dt / 2, gamma, config, helper_data, registered_variables, 1)
        primitive_state = _evolve_state_along_axis(primitive_state, dx, dt / 2, gamma, config, helper_data, registered_variables, 2)
        primitive_state = _evolve_state_along_axis(primitive_state, dx, dt, gamma, config, helper_data, registered_variables, 3)
        primitive_state = _evolve_state_along_axis(primitive_state, dx, dt / 2, gamma, config, helper_data, registered_variables, 2)
        primitive_state = _evolve_state_along_axis(primitive_state, dx, dt / 2, gamma, config, helper_data, registered_variables, 1)
    else:
        raise ValueError("Dimensionality not supported.")

    return primitive_state


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _evolve_state(
    primitive_state: STATE_TYPE,
    dx: Union[float, Float[Array, ""]], dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig, helper_data: HelperData,
    registered_variables: RegisteredVariables
) -> STATE_TYPE:
    
    if config.mhd:

        if config.dimensionality > 1:

            # THIS IS VERY PRELIMINARY; THIS KIND OF SPLITTING SHOULD NOT HAPPEN
            # IN EVERY EVOLVE_STATE CALL
            registered_variables_gas = registered_variables._replace(num_vars = registered_variables.num_vars - 3)

            gas_state = jnp.zeros((registered_variables_gas.num_vars, *primitive_state.shape[1:]), dtype = jnp.float64)
            gas_state = primitive_state[:-3, ...]
            magnetic_field = primitive_state[-3:, ...]

            # evolve gas state by half a time step
            evolved_gas = _evolve_gas_state(gas_state, dx, dt / 2, gamma, config, helper_data, registered_variables_gas)

            magnetic_field, evolved_gas = magnetic_update(magnetic_field, evolved_gas, dx, dt, registered_variables, config)

            # evolve gas state by half a time step
            evolved_gas = _evolve_gas_state(evolved_gas, dx, dt / 2, gamma, config, helper_data, registered_variables_gas)

            return jnp.concatenate((evolved_gas, magnetic_field), axis = 0)
        else:
            # error
            raise ValueError("MHD currently not supported in 1D.")

    else:
        return _evolve_gas_state(primitive_state, dx, dt, gamma, config, helper_data, registered_variables)