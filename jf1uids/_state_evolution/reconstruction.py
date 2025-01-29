from typing import Union
from jf1uids._geometry.geometry import CARTESIAN, STATE_TYPE
from jf1uids._state_evolution.limited_gradients import _calculate_limited_gradients, _calculate_limited_gradients3D
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.fluid import speed_of_sound
from jf1uids.fluid_equations.registered_variables import RegisteredVariables


import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped


from functools import partial


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