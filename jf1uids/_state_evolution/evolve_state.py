import jax.numpy as jnp
import jax

from functools import partial

from jf1uids._geometry.geometric_terms import _pressure_nozzling_source
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import gas_pressure_from_primitives_with_crs
from jf1uids._state_evolution.reconstruction import _reconstruct_at_interface
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids._geometry.boundaries import _boundary_handler, _boundary_handler3D
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive3D, primitive_state_from_conserved, primitive_state_from_conserved3D, conserved_state_from_primitive
from jf1uids._geometry.geometry import CARTESIAN
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids._riemann_solver.hll import _hll_solver, _hll_solver3D

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from jax.experimental import checkify

from typing import Union

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _get_conservative_derivative(
    conservative_states: Float[Array, "num_vars num_cells"],
    dx: Union[float, Float[Array, ""]],
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables
) -> Float[Array, "num_vars num_cells"]:
    """
    Time derivative of the conserved variables.

    Args:
        conservative_states: The conservative state array.
        dx: The cell width.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.

    Returns:
        The time derivative of the conserved variables.
    """
    primitive_states = primitive_state_from_conserved(conservative_states, gamma, registered_variables)

    primitive_states = _boundary_handler(primitive_states, config.left_boundary, config.right_boundary)
    
    # initialize the conservative derivative
    conservative_deriv = jnp.zeros_like(conservative_states)

    # get the left and right states at the interfaces
    primitives_left_of_interface, primitives_right_of_interface = _reconstruct_at_interface(primitive_states, dt, gamma, config, helper_data, registered_variables, axis = 1)

    # calculate the fluxes at the interfaces
    fluxes = _hll_solver(primitives_left_of_interface, primitives_right_of_interface, gamma, registered_variables)

    # update the conserved variables using the fluxes
    if config.geometry == CARTESIAN:
        conservative_deriv = conservative_deriv.at[:, config.num_ghost_cells:-config.num_ghost_cells].add(-1 / dx * (fluxes[:, 1:] - fluxes[:, :-1]))
    else:
        r = helper_data.geometric_centers
        rv = helper_data.volumetric_centers
        r_hat_alpha = helper_data.r_hat_alpha

        alpha = config.geometry

        r_plus_half = r.at[config.num_ghost_cells:-config.num_ghost_cells].add(dx / 2)
        r_minus_half = r.at[config.num_ghost_cells:-config.num_ghost_cells].add(-dx / 2)

        # calculate the source terms
        nozzling_source = _pressure_nozzling_source(primitive_states, config, helper_data, registered_variables)

        # update the conserved variables using the fluxes and source terms
        conservative_deriv = conservative_deriv.at[:, config.num_ghost_cells:-config.num_ghost_cells].add(1 / r_hat_alpha[config.num_ghost_cells:-config.num_ghost_cells] * (
            - (r_plus_half[config.num_ghost_cells:-config.num_ghost_cells] ** alpha * fluxes[:, 1:] - r_minus_half[config.num_ghost_cells:-config.num_ghost_cells] ** alpha * fluxes[:, :-1]) / dx
            + nozzling_source[:, 1:-1]
        ))

    return conservative_deriv

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _evolve_state(
    primitive_states: Float[Array, "num_vars num_cells"],
    dx: Union[float, Float[Array, ""]],
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables
) -> Float[Array, "num_vars num_cells"]:
    
    """Evolve the primitive state array.

    Args:
        primitive_states: The primitive state array.
        dx: The cell width.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.

    Returns:
        The evolved primitive state array.
    """

    # get the conserved variables
    conservative_states = conserved_state_from_primitive(primitive_states, gamma, registered_variables)

    # ===================== euler time step =====================

    # get the time derivative of the conservative variables
    conservative_deriv = _get_conservative_derivative(conservative_states, dx, dt, gamma, config, helper_data, registered_variables)
    
    # update the conservative variables, here with an Euler step
    conservative_states = conservative_states + dt * conservative_deriv

    # ===========================================================

    # update the primitive variables
    primitive_states = primitive_state_from_conserved(conservative_states, gamma, registered_variables)

    return primitive_states

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables', 'axis'])
def _evolve_state3D_in_one_dimension(
    primitive_states: Float[Array, "num_vars num_cells num_cells num_cells"],
    dx: Union[float, Float[Array, ""]],
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    axis: int
) -> Float[Array, "num_vars num_cells num_cells num_cells"]:
    
    primitive_states = _boundary_handler3D(primitive_states, config.first_order_fallback)

    # get conserved variables
    conservative_states = conserved_state_from_primitive3D(primitive_states, gamma, registered_variables)

    num_cells = primitive_states.shape[axis]

    # flux in x-direction
    if config.first_order_fallback:
        primitive_states_left = jax.lax.slice_in_dim(primitive_states, 0, num_cells - 1, axis = axis)
        primitive_states_right = jax.lax.slice_in_dim(primitive_states, 1, num_cells, axis = axis)
    else:
        primitive_states_left, primitive_states_right = _reconstruct_at_interface(primitive_states, dt, gamma, config, helper_data, registered_variables, axis)
    
    fluxes_x = _hll_solver3D(primitive_states_left, primitive_states_right, gamma, registered_variables, axis)

    flux_length = fluxes_x.shape[axis]

    # update the conserved variables
    conserved_change = -1 / dx * (jax.lax.slice_in_dim(fluxes_x, 1, flux_length, axis = axis) - jax.lax.slice_in_dim(fluxes_x, 0, flux_length - 1, axis = axis)) * dt

    if axis == 1:
        conservative_states = conservative_states.at[:, config.num_ghost_cells:-config.num_ghost_cells, :, :].add(conserved_change)
    elif axis == 2:
        conservative_states = conservative_states.at[:, :, config.num_ghost_cells:-config.num_ghost_cells, :].add(conserved_change)
    elif axis == 3:
        conservative_states = conservative_states.at[:, :, :, config.num_ghost_cells:-config.num_ghost_cells].add(conserved_change)
    else:
        raise ValueError("Invalid axis")

    primitive_states = primitive_state_from_conserved3D(conservative_states, gamma, registered_variables)
    primitive_states = _boundary_handler3D(primitive_states, config.first_order_fallback)

    # check if the pressure is still positive
    p = primitive_states[registered_variables.pressure_index]
    rho = primitive_states[registered_variables.density_index]

    if config.runtime_debugging:
        checkify.check(jnp.all(p >= 0), "pressure needs to be non-negative, minimum pressure {pmin} at index {index}", pmin=jnp.min(p), index=jnp.unravel_index(jnp.argmin(p), p.shape))
        checkify.check(jnp.all(rho >= 0), "density needs to be non-negative, minimum density {rhomin} at index {index}", rhomin=jnp.min(rho), index=jnp.unravel_index(jnp.argmin(rho), rho.shape))
    

    return primitive_states


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _evolve_state3D(
    primitive_states: Float[Array, "num_vars num_cells num_cells num_cells"],
    dx: Union[float, Float[Array, ""]], dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig, helper_data: HelperData,
    registered_variables: RegisteredVariables
) -> Float[Array, "num_vars num_cells num_cells num_cells"]:
    """Evolve the primitive state array.

    Args:
        primitive_states: The primitive state array.
        dx: The cell width.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.

    Returns:
        The evolved primitive state array.
    """

    if config.first_order_fallback:
        primitive_states = _evolve_state3D_in_one_dimension(primitive_states, dx, dt, gamma, config, helper_data, registered_variables, 1)
        primitive_states = _evolve_state3D_in_one_dimension(primitive_states, dx, dt, gamma, config, helper_data, registered_variables, 2)
        primitive_states = _evolve_state3D_in_one_dimension(primitive_states, dx, dt, gamma, config, helper_data, registered_variables, 3)
    else:
        # advance in x by dt/2 -> y by dt/2 -> z by dt -> y by dt/2 -> x by dt/2
        primitive_states = _evolve_state3D_in_one_dimension(primitive_states, dx, dt / 2, gamma, config, helper_data, registered_variables, 1)
        primitive_states = _evolve_state3D_in_one_dimension(primitive_states, dx, dt / 2, gamma, config, helper_data, registered_variables, 2)
        primitive_states = _evolve_state3D_in_one_dimension(primitive_states, dx, dt, gamma, config, helper_data, registered_variables, 3)
        primitive_states = _evolve_state3D_in_one_dimension(primitive_states, dx, dt / 2, gamma, config, helper_data, registered_variables, 2)
        primitive_states = _evolve_state3D_in_one_dimension(primitive_states, dx, dt / 2, gamma, config, helper_data, registered_variables, 1)

    return primitive_states