import jax.numpy as jnp
import jax

from functools import partial

from jf1uids.geometry.boundaries import _boundary_handler
from jf1uids.fluid_equations.fluid import conserved_state, primitive_state_from_conserved, speed_of_sound
from jf1uids.geometry.geometry import CARTESIAN
from jf1uids.spatial_reconstruction.limiters import _minmod
from jf1uids.riemann_solver.hll import _hll_solver

@partial(jax.jit, static_argnames=['geometry'])
def calculate_limited_gradients(primitive_states, dx, geometry, rv):
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

@partial(jax.jit, static_argnames=['geometry', 'first_order_fallback'])
def reconstruct_at_interface(primitive_states, dt, dx, gamma, geometry, first_order_fallback, helper_data):

    # get fluid variables for convenience
    rho, u, p = primitive_states

    if geometry == CARTESIAN:
        distances_to_left_interfaces = dx / 2 # distances r_i - r_{i-1/2}
        distances_to_right_interfaces = dx / 2 # distances r_{i+1/2} - r_i
    else:
        r = helper_data.geometric_centers
        rv = helper_data.volumetric_centers

        distances_to_left_interfaces = rv[1:-1] - (r[1:-1] - dx / 2)
        distances_to_right_interfaces = (r[1:-1] + dx / 2) - rv[1:-1]

    # get the limited gradients on the cells
    limited_gradients = calculate_limited_gradients(primitive_states, dx, geometry, helper_data.volumetric_centers)

    # fallback to 1st order
    if first_order_fallback:
        limited_gradients = jnp.zeros_like(limited_gradients)

    # calculate the sound speed
    c = speed_of_sound(rho, p, gamma)

    # calculate the vectors making up A_W
    A_W_1 = jnp.stack([u, jnp.zeros_like(u), jnp.zeros_like(u)], axis = 0)
    A_W_2 = jnp.stack([rho, u, rho * c ** 2], axis = 0)
    A_W_3 = jnp.stack([jnp.zeros_like(u), 1 / rho, u], axis = 0)
    # maybe better construct A_W as a 3x3xN tensor
    # A_W = jnp.stack([A_W_1, A_W_2, A_W_3], axis = 0)

    projected_gradients = A_W_1[:, 1:-1] * limited_gradients[0, :] + A_W_2[:, 1:-1] * limited_gradients[1, :] + A_W_3[:, 1:-1] * limited_gradients[2, :]

    # predictor step
    predictors = primitive_states.at[:, 1:-1].add(-dt / 2 * projected_gradients)

    # compute primitives at the interfaces
    primitives_left = predictors.at[:, 1:-1].add(-distances_to_left_interfaces * limited_gradients)
    primitives_right = predictors.at[:, 1:-1].add(distances_to_right_interfaces * limited_gradients)

    # the first entries are the state to the left and right
    # of the interface between cell 1 and 2
    return primitives_right[:, 1:-2], primitives_left[:, 2:-1]

@partial(jax.jit, static_argnames=['geometry'])
def pressure_nozzling_source(primitive_states, dx, r, rv, r_hat_alpha, geometry):
    _, _, p = primitive_states

    # calculate the limited gradients on the cells
    _, _, dp_dr = calculate_limited_gradients(primitive_states, dx, geometry, rv)

    pressure_nozzling = r[1:-1] ** (geometry - 1) * p[1:-1] + (r_hat_alpha[1:-1] - rv[1:-1] * r[1:-1] ** (geometry - 1)) * dp_dr

    return jnp.stack([jnp.zeros_like(pressure_nozzling), geometry * pressure_nozzling, jnp.zeros_like(pressure_nozzling)], axis = 0)

@partial(jax.jit, static_argnames=['config'])
def get_conservative_derivative(conservative_states, dx, dt, gamma, config, helper_data):
    """
    Time derivative of the conserved variables
    """
    primitive_states = primitive_state_from_conserved(conservative_states, gamma)

    primitive_states = _boundary_handler(primitive_states, config.left_boundary, config.right_boundary)
    
    # initialize the conservative derivative
    conservative_deriv = jnp.zeros_like(conservative_states)

    # get the left and right states at the interfaces
    primitives_left_of_interface, primitives_right_of_interface = reconstruct_at_interface(primitive_states, dt, dx, gamma, config.geometry, config.first_order_fallback, helper_data)

    # calculate the fluxes at the interfaces
    fluxes = _hll_solver(primitives_left_of_interface, primitives_right_of_interface, gamma)

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
        nozzling_source = pressure_nozzling_source(primitive_states, dx, r, rv, r_hat_alpha, config.geometry)

        # update the conserved variables using the fluxes and source terms
        conservative_deriv = conservative_deriv.at[:, config.num_ghost_cells:-config.num_ghost_cells].add(1 / r_hat_alpha[config.num_ghost_cells:-config.num_ghost_cells] * (
            - (r_plus_half[config.num_ghost_cells:-config.num_ghost_cells] ** alpha * fluxes[:, 1:] - r_minus_half[config.num_ghost_cells:-config.num_ghost_cells] ** alpha * fluxes[:, :-1]) / dx
            + nozzling_source[:, 1:-1]
        ))

    return conservative_deriv

@partial(jax.jit, static_argnames=['config'])
def evolve_state(primitive_states, dx, dt, gamma, config, helper_data):

    # get the conserved variables
    conservative_states = conserved_state(primitive_states, gamma)

    # ===================== euler time step =====================

    # get the time derivative of the conservative variables
    conservative_deriv = get_conservative_derivative(conservative_states, dx, dt, gamma, config, helper_data)
    
    # update the conservative variables, here with an Euler step
    conservative_states = conservative_states + dt * conservative_deriv

    # ===========================================================

    # update the primitive variables
    primitive_states = primitive_state_from_conserved(conservative_states, gamma)

    return primitive_states