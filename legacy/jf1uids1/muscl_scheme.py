import time
import jax.numpy as jnp
from jf1uids.fluid import conserved_state, primitive_state_from_conserved, speed_of_sound
from jf1uids.limiters import minmod, superbee
from jf1uids.riemann import hll_solver

def reconstruct_at_interface(primitive_states, dt, dx, gamma):

    # get fluid variables for convenience
    rho, u, p = primitive_states

    # get the limited gradients on the cells
    limited_gradients = minmod(primitive_states[:, 1:-1] - primitive_states[:, :-2], primitive_states[:, 2:] - primitive_states[:, 1:-1])

    # fallback to 1st order
    # limited_gradients = jnp.zeros_like(limited_gradients)

    # advance the primitive states of the cells by half a timestep
    # predictors = primitives - dt / (2 * dx) * A_W * limited_gradients

    # calculate the sound speed
    c = speed_of_sound(rho, p, gamma)

    # calculate the vectors making up A_W
    A_W_1 = jnp.stack([u, jnp.zeros_like(u), jnp.zeros_like(u)], axis = 0)
    A_W_2 = jnp.stack([rho, u, rho * c ** 2], axis = 0)
    A_W_3 = jnp.stack([jnp.zeros_like(u), 1 / rho, u], axis = 0)
    # better construct A_W as a 3x3xN tensor
    # A_W = jnp.stack([A_W_1, A_W_2, A_W_3], axis = 0)

    # print the shape of limited_gradients
    # print(limited_gradients.shape)

    # print the shape of A_W
    # print(A_W.shape)

    # predictor step
    predictors = primitive_states.at[:, 1:-1].add(-dt / (2 * dx) * (A_W_1[:, 1:-1] * limited_gradients[0, :] + A_W_2[:, 1:-1] * limited_gradients[1, :] + A_W_3[:, 1:-1] * limited_gradients[2, :]))

    # compute primitives at the interfaces
    primitives_left = predictors.at[:, 1:-1].add(-1/2 * limited_gradients)
    primitives_right = predictors.at[:, 1:-1].add(1/2 * limited_gradients)

    # the first entries are the state to the left and right
    # of the interface between cell 1 and 2
    return primitives_right[:, 1:-2], primitives_left[:, 2:-1]

def evolve_state(primitive_states, dx, dt, gamma):
    """
    Evolve the state by dt.
    """

    # print("primitive states")
    # print(primitive_states)
    
    # calculate conserved variables
    conserved_states = conserved_state(primitive_states, gamma)

    # get the left and right states at the interfaces
    primitives_left_of_interface, primitives_right_of_interface = reconstruct_at_interface(primitive_states, dt, dx, gamma)

    # print("primitives left")
    # print(primitives_left_of_interface)
    # print("primitives right")
    # print(primitives_right_of_interface)

    # calculate the fluxes at the interfaces
    fluxes = hll_solver(primitives_left_of_interface, primitives_right_of_interface, gamma)

    # print("conserved states")
    # print(conserved_states)

    # print("fluxes")
    # print(-dt / dx *  (fluxes[:, 1:] - fluxes[:, :-1]))

    # time.sleep(10)

    # update the conserved variables using the fluxes
    conserved_states = conserved_states.at[:, 2:-2].add(-dt / dx * (fluxes[:, 1:] - fluxes[:, :-1]))

    # return the updated primitive variables
    return primitive_state_from_conserved(conserved_states, gamma)