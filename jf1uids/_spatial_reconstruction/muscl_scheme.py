import jax.numpy as jnp
import jax

from functools import partial

from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import gas_pressure_from_primitives_with_crs
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids._geometry.boundaries import _boundary_handler, _boundary_handler3D
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive3D, primitive_state_from_conserved, primitive_state_from_conserved3D, speed_of_sound, conserved_state_from_primitive
from jf1uids._geometry.geometry import CARTESIAN, STATE_TYPE
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids._spatial_reconstruction.limiters import _minmod
from jf1uids._riemann_solver.hll import _hll_solver, _hll_solver3D

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from jax.experimental import checkify

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
    epsilon = 1e-11  # Small constant to prevent division by zero
    a = (primitive_states[:, 1:-1] - primitive_states[:, :-2]) / cell_distances_left
    b = (primitive_states[:, 2:] - primitive_states[:, 1:-1]) / cell_distances_right
    g = jnp.where(
        jnp.abs(a) > epsilon,  # Avoid division if `a` is very small
        b / (a + epsilon),  # Add epsilon to `a` for numerical stability
        jnp.zeros_like(a)
    )
    # slope_limited = jnp.maximum(0, jnp.minimum(1, g))  # Minmod limiter
    slope_limited = jnp.maximum(0, jnp.minimum(1.3, g))  # Osher limiter with beta = 1.3
    # ospre limiter
    # slope_limited = (1.5 * (g ** 2 + g)) / (g ** 2 + g + 1)
    limited_gradients = slope_limited * a

    # # formulation 2:
    # limited_gradients = _minmod(
    #     (primitive_states[:, 1:-1] - primitive_states[:, :-2]) / cell_distances_left,
    #     (primitive_states[:, 2:] - primitive_states[:, 1:-1]) / cell_distances_right
    # )

    return limited_gradients

# TODO: improve shape annotations, two smaller in the flux_direction_index dimension
# or maybe better: equal shapes everywhere
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['axis'])
def _calculate_limited_gradients3D(primitive_states: STATE_TYPE, dx: Union[float, Float[Array, ""]], axis: int) -> Float[Array, "num_vars num_cells_a num_cells_b num_cells_c"]:
    # axis = 1 for x (0th axis are the primitive variables)

    # formulation 1:
    epsilon = 1e-11  # Small constant to prevent division by zero

    # get array sizee along the axis
    num_cells = primitive_states.shape[axis]
    
    a = (jax.lax.slice_in_dim(primitive_states, 1, num_cells - 1, axis = axis) - jax.lax.slice_in_dim(primitive_states, 0, num_cells - 2, axis = axis)) / dx
    b = (jax.lax.slice_in_dim(primitive_states, 2, num_cells, axis = axis) - jax.lax.slice_in_dim(primitive_states, 1, num_cells - 1, axis = axis)) / dx
    # g = jnp.where(
    #     jnp.abs(a) > epsilon,  # Avoid division if `a` is very small
    #     b / (a + epsilon),  # Add epsilon to `a` for numerical stability
    #     jnp.zeros_like(a)
    # )
    # # slope_limited = jnp.maximum(0, jnp.minimum(1, g))  # Minmod limiter
    # slope_limited = jnp.maximum(0, jnp.minimum(1.3, g))  # Osher limiter with beta = 1.3
    # # ospre limiter
    # # slope_limited = (1.5 * (g ** 2 + g)) / (g ** 2 + g + 1)
    # limited_gradients = slope_limited * a

    limited_gradients = _minmod(a, b)

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
@partial(jax.jit, static_argnames=['registered_variables', 'axis'])
def _reconstruct_at_interface3D(primitive_states: STATE_TYPE, dt: Union[float, Float[Array, ""]], dx: Union[float, Float[Array, ""]], gamma: Union[float, Float[Array, ""]], registered_variables: RegisteredVariables, axis: int) -> tuple[Float[Array, "num_vars num_cells_a num_cells_b num_cells_c"], Float[Array, "num_vars num_cells_a num_cells_b num_cells_c"]]:
    """Limited linear reconstruction of the primitive variables at the interfaces.

    Args:
        primitive_states: The primitive state array.
        dt: The time step.
        dx: The cell width.
        gamma: The adiabatic index.

    Returns:
        The primitive variables at both sides of the interfaces.
    """

    num_cells = primitive_states.shape[axis]

    # get fluid variables for convenience
    rho = primitive_states[registered_variables.density_index]
    p = primitive_states[registered_variables.pressure_index]
    u = primitive_states[axis]

    # get the limited gradients on the cells
    limited_gradients = _calculate_limited_gradients3D(primitive_states, dx, axis)

    # calculate the sound speed
    c = speed_of_sound(rho, p, gamma)

    # see https://diglib.uibk.ac.at/download/pdf/4422963.pdf, 2.11

    # calculate the vectors making up A_W
    A_W = jnp.zeros((registered_variables.num_vars, registered_variables.num_vars, num_cells, num_cells, num_cells))
    
    # set u diagonal
    A_W = A_W.at[jnp.arange(5), jnp.arange(5)].set(u)

    # set rest
    A_W = A_W.at[axis, 0].set(rho)
    A_W = A_W.at[4, 1].set(rho * c ** 2)
    A_W = A_W.at[axis, 4].set(1 / rho)

    A_W = jax.lax.slice_in_dim(A_W, 1, num_cells - 1, axis = axis + 1)

    # maybe write using tensordot
    projected_gradients = jnp.einsum('baxyz, axyz -> bxyz', A_W, limited_gradients)

    # predictor step
    predictors = jax.lax.slice_in_dim(primitive_states, 1, num_cells - 1, axis = axis) - dt / 2 * projected_gradients

    # compute primitives at the interfaces
    primitives_left = predictors - dx/2 * limited_gradients
    primitives_right = predictors + dx/2 * limited_gradients

    # the first entries are the state to the left and right
    # of the interface between cell 1 and 2
    num_prim = primitives_left.shape[axis]
    return jax.lax.slice_in_dim(primitives_right, 0, num_prim - 1, axis = axis), jax.lax.slice_in_dim(primitives_left, 1, num_prim, axis = axis)

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

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables', 'axis'])
def _evolve_state3D_in_one_dimension(primitive_states: Float[Array, "num_vars num_cells num_cells num_cells"], dx: Union[float, Float[Array, ""]], dt: Float[Array, ""], gamma: Union[float, Float[Array, ""]], config: SimulationConfig, helper_data: HelperData, registered_variables: RegisteredVariables, axis: int) -> Float[Array, "num_vars num_cells num_cells num_cells"]:
    
    primitive_states = _boundary_handler3D(primitive_states, config.first_order_fallback)

    # get conserved variables
    conservative_states = conserved_state_from_primitive3D(primitive_states, gamma, registered_variables)

    num_cells = primitive_states.shape[axis]

    # flux in x-direction
    if config.first_order_fallback:
        primitive_states_left = jax.lax.slice_in_dim(primitive_states, 0, num_cells - 1, axis = axis)
        primitive_states_right = jax.lax.slice_in_dim(primitive_states, 1, num_cells, axis = axis)
    else:
        primitive_states_left, primitive_states_right = _reconstruct_at_interface3D(primitive_states, dt, dx, gamma, registered_variables, axis)
    
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
def _evolve_state3D(primitive_states: Float[Array, "num_vars num_cells num_cells num_cells"], dx: Union[float, Float[Array, ""]], dt: Float[Array, ""], gamma: Union[float, Float[Array, ""]], config: SimulationConfig, helper_data: HelperData, registered_variables: RegisteredVariables) -> Float[Array, "num_vars num_cells num_cells num_cells"]:
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