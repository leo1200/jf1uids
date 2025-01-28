import jax
from functools import partial

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

CARTESIAN = 0
CYLINDRICAL = 1
SPHERICAL = 2

STATE_TYPE = Union[Float[Array, "num_vars num_cells_x"], Float[Array, "num_vars num_cells_x num_cells_y"], Float[Array, "num_vars num_cells_x num_cells_y num_cells_z"]]

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['geometry'])
def _r_hat_alpha(r: Float[Array, "num_cells"], dr: Union[float, Float[Array, ""]], geometry: int) -> Float[Array, "num_cells"]:
    if geometry == SPHERICAL:
        return r ** 2 + 1/12 * dr ** 2
    elif geometry == CYLINDRICAL:
        return r
    else:
        raise ValueError("Unknown geometry / not for cartesian coordinates")
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['geometry'])
def _center_of_volume(r: Float[Array, "num_cells"], dr: Union[float, Float[Array, ""]], geometry: int) -> Float[Array, "num_cells"]:
    if geometry == CYLINDRICAL:
        return (r ** 2 + 1/12 * dr ** 2) / r ** 2 * r
    elif geometry == SPHERICAL:
        r_hat = _r_hat_alpha(r, dr, geometry)
        return (r ** 2 + 1/4 * dr ** 2) / r_hat * r
    