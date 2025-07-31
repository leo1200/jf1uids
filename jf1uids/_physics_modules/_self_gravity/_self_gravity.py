"""
Fourier-based Poisson solver and simple source term handling
of self gravity. To be improved to an energy-conserving scheme.
"""

# general
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Union

# fft, in the future use
# https://github.com/DifferentiableUniverseInitiative/JaxPM

# jf1uids data classes
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from jf1uids._physics_modules._self_gravity._poisson_solver import _compute_gravitational_potential
from jf1uids._riemann_solver._riemann_solver import _riemann_solver
from jf1uids._stencil_operations._stencil_operations import _stencil_add
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import CONSERVATIVE_SOURCE_TERM, SIMPLE_SOURCE_TERM, STATE_TYPE_ALTERED, SimulationConfig

# jf1uids constants
from jf1uids.option_classes.simulation_config import FIELD_TYPE, HLL, HLLC, OPEN_BOUNDARY, STATE_TYPE

# jf1uids functions
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids._riemann_solver.hll import _hll_solver, _hllc_solver
from jf1uids._state_evolution.reconstruction import _reconstruct_at_interface_split
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, primitive_state_from_conserved, speed_of_sound


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis', 'grid_spacing', 'registered_variables', 'config'])
def _gravitational_source_term_along_axis(
        gravitational_potential: FIELD_TYPE,
        primitive_state: STATE_TYPE,
        grid_spacing: float,
        registered_variables: RegisteredVariables,
        dt: Union[float, Float[Array, ""]],
        gamma: Union[float, Float[Array, ""]],
        config: SimulationConfig,
        helper_data: HelperData,
        axis: int,
) -> STATE_TYPE:
    
    """
    Compute the source term for the self-gravity solver along a single axis.
    Currently, simply density * gravitational_acceleration for the momentum 
    and density * velocity * gravitational_acceleration for the energy.

    Args:
        gravitational_potential: The gravitational potential.
        primitive_state: The primitive state.
        grid_spacing: The grid spacing.
        registered_variables: The registered variables.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.
        axis: The axis along which to compute the source term.

    Returns:
        The source term.
    
    """

    rho = primitive_state[registered_variables.density_index]
    v_axis = primitive_state[axis]

    # a_i = - (phi_{i+1} - phi_{i-1}) / (2 * dx)
    acceleration = -_stencil_add(gravitational_potential, indices = (1, -1), factors = (1.0, -1.0), axis = axis - 1) / (2 * grid_spacing)
    # it is axis - 1 because the axis is 1-indexed as usually the zeroth axis are the different
    # fields in the state vector not the spatial dimensions, but here we only have the spatial dimensions

    source_term = jnp.zeros_like(primitive_state)

    # set momentum source
    source_term = source_term.at[axis].set(rho * acceleration)

    if config.self_gravity_version == SIMPLE_SOURCE_TERM:

        # set energy source
        source_term = source_term.at[registered_variables.pressure_index].set(rho * v_axis * acceleration)

    elif config.self_gravity_version == CONSERVATIVE_SOURCE_TERM:
        # ===============================================

        # better energy source
        if config.first_order_fallback:
            primitive_state_left = jnp.roll(primitive_state, shift = 1, axis = axis)
            primitive_state_right = primitive_state
        else:
            primitive_state_left, primitive_state_right = _reconstruct_at_interface_split(primitive_state, dt, gamma, config, helper_data, registered_variables, axis)
        
        # at index i, the fluxes array contains the flux from i-1 to i
        fluxes = _riemann_solver(primitive_state_left, primitive_state_right, primitive_state, gamma, config, registered_variables, axis)
        fluxes_i_to_ip1 = jnp.maximum(jnp.roll(fluxes, shift = -1, axis = axis), 0)
        fluxes_i_to_im1 = jnp.minimum(fluxes, 0)

        acc_backward = -_stencil_add(gravitational_potential, indices = (0, -1), factors = (1.0, -1.0), axis = axis - 1) / grid_spacing
        acc_forward = -_stencil_add(gravitational_potential, indices = (1, 0), factors = (1.0, -1.0), axis = axis - 1) / grid_spacing

        fluxes_acc = fluxes_i_to_im1 * acc_backward + fluxes_i_to_ip1 * acc_forward

        source_term = source_term.at[registered_variables.pressure_index].set(fluxes_acc[0])

        # ===============================================

    return source_term

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _apply_self_gravity(
    primitive_state: STATE_TYPE,
    old_primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    gamma: Union[float, Float[Array, ""]],
    gravitational_constant: Union[float, Float[Array, ""]],
    dt: Union[float, Float[Array, ""]]
) -> STATE_TYPE:

    rho = old_primitive_state[registered_variables.density_index]

    potential = _compute_gravitational_potential(rho, config.grid_spacing, config, gravitational_constant)

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
                                        helper_data,
                                        i + 1
                                    )

    conserved_state = conserved_state_from_primitive(primitive_state, gamma, config, registered_variables)

    conserved_state = conserved_state + dt * source_term

    primitive_state = primitive_state_from_conserved(conserved_state, gamma, config, registered_variables)

    primitive_state = _boundary_handler(primitive_state, config)

    return primitive_state