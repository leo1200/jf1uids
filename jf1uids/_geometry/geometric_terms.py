from typing import Union
from jf1uids._state_evolution.limited_gradients import _calculate_limited_gradients
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from functools import partial

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