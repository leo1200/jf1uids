import jax
from functools import partial

CARTESIAN = 0
CYLINDRICAL = 1
SPHERICAL = 2

@partial(jax.jit, static_argnames=['geometry'])
def _r_hat_alpha(r, dr, geometry):
    if geometry == SPHERICAL:
        return r ** 2 + 1/12 * dr ** 2
    elif geometry == CYLINDRICAL:
        return r
    else:
        raise ValueError("Unknown geometry / not for cartesian coordinates")
    
@partial(jax.jit, static_argnames=['geometry'])
def _center_of_volume(r, dr, geometry):
    if geometry == CYLINDRICAL:
        return (r ** 2 + 1/12 * dr ** 2) / r ** 2 * r
    elif geometry == SPHERICAL:
        r_hat = _r_hat_alpha(r, dr, geometry)
        return (r ** 2 + 1/4 * dr ** 2) / r_hat * r
    