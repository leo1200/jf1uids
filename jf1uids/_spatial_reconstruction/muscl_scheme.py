import jax.numpy as jnp
import jax

from functools import partial

from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import gas_pressure_from_primitives_with_crs
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids.fluid_equations.fluid import primitive_state_from_conserved, speed_of_sound, conserved_state_from_primitive
from jf1uids._geometry.geometry import CARTESIAN
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids._spatial_reconstruction.limiters import _minmod
from jf1uids._riemann_solver.hll import _hll_solver

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['geometry'])
def _calculate_limited_gradients(primitive_states: Float[Array, "num_vars num_cells"], dx: Union[float, Float[Array, ""]], geometry: int, rv: Float[Array, "num_cells"]) -> Float[Array, "num_vars num_cells-2"]:
    """
    Calculate the limited gradients of the primitive variables.

    Args:
        primitive_states: The primitive state array.
        dx: The cell width.
        geometry: The geometry of the domain.
        rv: The volumetric centers of the cells.

    Returns:
        The limited gradients of the primitive variables.

    """
    if geometry == CARTESIAN:
        cell_distances_left = dx # distances r_i - r_{i-1}
        cell_distances_right = dx # distances r_{i+1} - r_i
    else:
        # calculate the distances
        cell_distances_left = rv[1:-1] - rv[:-2]
        cell_distances_right = rv[2:] - rv[1:-1]

    # formulation 1:
    # a = (primitive_states[:, 1:-1] - primitive_states[:, :-2]) / cell_distances_left
    # b = (primitive_states[:, 2:] - primitive_states[:, 1:-1]) / cell_distances_right
    # g = jnp.where(a != 0, jnp.divide(b, a), jnp.zeros_like(a))
    # slope_limited = jnp.maximum(0, jnp.minimum(1, g)) # minmod
    # limited_gradients = slope_limited * a

    # formulation 2:
    limited_gradients = _minmod(
        (primitive_states[:, 1:-1] - primitive_states[:, :-2]) / cell_distances_left,
        (primitive_states[:, 2:] - primitive_states[:, 1:-1]) / cell_distances_right
    )
    
    return limited_gradients

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['geometry', 'first_order_fallback', 'registered_variables'])
def _reconstruct_at_interface(primitive_states: Float[Array, "num_vars num_cells"], dt: Float[Array, ""], dx: Union[float, Float[Array, ""]], gamma: Union[float, Float[Array, ""]], geometry: int, first_order_fallback: bool, helper_data: HelperData, registered_variables: RegisteredVariables) -> tuple[Float[Array, "num_vars num_interfaces"], Float[Array, "num_vars num_interfaces"]]:
    """Limited linear reconstruction of the primitive variables at the interfaces.

    Args:
        primitive_states: The primitive state array.
        dt: The time step.
        dx: The cell width.
        gamma: The adiabatic index.
        geometry: The geometry of the domain.
        first_order_fallback: Fallback to no linear reconstruction, 1st order Godunov scheme.
        helper_data: The helper data.

    Returns:
        The primitive variables at both sides of the interfaces.
    """

    # get fluid variables for convenience
    rho = primitive_states[registered_variables.density_index]
    u = primitive_states[registered_variables.velocity_index]

    # if registered_variables.cosmic_ray_n_active:
    #     p = gas_pressure_from_primitives_with_crs(primitive_states, registered_variables)
    # else:
    #     p = primitive_states[registered_variables.pressure_index]

    p = primitive_states[registered_variables.pressure_index]

    if geometry == CARTESIAN:
        distances_to_left_interfaces = dx / 2 # distances r_i - r_{i-1/2}
        distances_to_right_interfaces = dx / 2 # distances r_{i+1/2} - r_i
    else:
        r = helper_data.geometric_centers
        rv = helper_data.volumetric_centers

        distances_to_left_interfaces = rv[1:-1] - (r[1:-1] - dx / 2)
        distances_to_right_interfaces = (r[1:-1] + dx / 2) - rv[1:-1]

    # get the limited gradients on the cells
    limited_gradients = _calculate_limited_gradients(primitive_states, dx, geometry, helper_data.volumetric_centers)

    # fallback to 1st order
    if first_order_fallback:
        limited_gradients = jnp.zeros_like(limited_gradients)

    # calculate the sound speed
    c = speed_of_sound(rho, p, gamma)

    # calculate the vectors making up A_W
    A_W_1 = jnp.zeros_like(primitive_states)
    A_W_1 = A_W_1.at[registered_variables.density_index].set(u)

    A_W_2 = jnp.zeros_like(primitive_states)
    A_W_2 = A_W_2.at[registered_variables.density_index].set(rho)
    A_W_2 = A_W_2.at[registered_variables.velocity_index].set(u)
    A_W_2 = A_W_2.at[registered_variables.pressure_index].set(rho * c ** 2)

    A_W_3 = jnp.zeros_like(primitive_states)
    A_W_3 = A_W_3.at[registered_variables.velocity_index].set(1 / rho)
    A_W_3 = A_W_3.at[registered_variables.pressure_index].set(u)

    # TODO: generalize this for more than 3 variables
    projected_gradients = A_W_1[:, 1:-1] * limited_gradients[0, :] + A_W_2[:, 1:-1] * limited_gradients[1, :] + A_W_3[:, 1:-1] * limited_gradients[2, :]

    if registered_variables.wind_density_active:
        A_W_wind = jnp.zeros_like(primitive_states)
        A_W_wind = A_W_wind.at[registered_variables.wind_density_index].set(u)
        projected_gradients += A_W_wind[:, 1:-1] * limited_gradients[registered_variables.wind_density_index, :]

    if registered_variables.cosmic_ray_n_active:
        A_W_CR = jnp.zeros_like(primitive_states)
        A_W_CR = A_W_CR.at[registered_variables.cosmic_ray_n_index].set(u)
        projected_gradients += A_W_CR[:, 1:-1] * limited_gradients[registered_variables.cosmic_ray_n_index, :]
    
    # predictor step
    predictors = primitive_states.at[:, 1:-1].add(-dt / 2 * projected_gradients)

    # compute primitives at the interfaces
    primitives_left = predictors.at[:, 1:-1].add(-distances_to_left_interfaces * limited_gradients)
    primitives_right = predictors.at[:, 1:-1].add(distances_to_right_interfaces * limited_gradients)

    # the first entries are the state to the left and right
    # of the interface between cell 1 and 2
    return primitives_right[:, 1:-2], primitives_left[:, 2:-1]

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['geometry', 'registered_variables'])
def _pressure_nozzling_source(primitive_states: Float[Array, "num_vars num_cells"], dx: Union[float, Float[Array, ""]], r: Float[Array, "num_cells"], rv: Float[Array, "num_cells"], r_hat_alpha: Float[Array, "num_cells"], geometry: int, registered_variables: RegisteredVariables) -> Float[Array, "num_vars num_cells-2"]:
    """Pressure nozzling source term as of the geometry of the domain.

    Args:
        primitive_states: The primitive state array.
        dx: The cell width.
        r: The geometric centers of the cells.
        rv: The volumetric centers of the cells.
        r_hat_alpha: The r_hat_alpha values.
        geometry: The geometry of the domain.

    Returns:
        The pressure nozzling source
    """
    p = primitive_states[registered_variables.pressure_index]

    # calculate the limited gradients on the cells
    dp_dr = _calculate_limited_gradients(primitive_states, dx, geometry, rv)[registered_variables.pressure_index]

    pressure_nozzling = r[1:-1] ** (geometry - 1) * p[1:-1] + (r_hat_alpha[1:-1] - rv[1:-1] * r[1:-1] ** (geometry - 1)) * dp_dr

    nozzling = jnp.zeros((registered_variables.num_vars, p.shape[0] - 2))
    nozzling = nozzling.at[registered_variables.velocity_index].set(geometry * pressure_nozzling)

    return nozzling


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _get_conservative_derivative(conservative_states: Float[Array, "num_vars num_cells"], dx: Union[float, Float[Array, ""]], dt: Float[Array, ""], gamma: Union[float, Float[Array, ""]], config: SimulationConfig, helper_data: HelperData, registered_variables: RegisteredVariables) -> Float[Array, "num_vars num_cells"]:
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
    primitives_left_of_interface, primitives_right_of_interface = _reconstruct_at_interface(primitive_states, dt, dx, gamma, config.geometry, config.first_order_fallback, helper_data, registered_variables)

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
        nozzling_source = _pressure_nozzling_source(primitive_states, dx, r, rv, r_hat_alpha, config.geometry, registered_variables)

        # update the conserved variables using the fluxes and source terms
        conservative_deriv = conservative_deriv.at[:, config.num_ghost_cells:-config.num_ghost_cells].add(1 / r_hat_alpha[config.num_ghost_cells:-config.num_ghost_cells] * (
            - (r_plus_half[config.num_ghost_cells:-config.num_ghost_cells] ** alpha * fluxes[:, 1:] - r_minus_half[config.num_ghost_cells:-config.num_ghost_cells] ** alpha * fluxes[:, :-1]) / dx
            + nozzling_source[:, 1:-1]
        ))

    return conservative_deriv

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables'])
def _evolve_state(primitive_states: Float[Array, "num_vars num_cells"], dx: Union[float, Float[Array, ""]], dt: Float[Array, ""], gamma: Union[float, Float[Array, ""]], config: SimulationConfig, helper_data: HelperData, registered_variables: RegisteredVariables) -> Float[Array, "num_vars num_cells"]:
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