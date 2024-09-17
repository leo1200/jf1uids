import jax
from functools import partial

CARTESIAN = 0
CYLINDRICAL = 1
SPHERICAL = 2

@partial(jax.jit, static_argnames=['alpha_geom'])
def _r_hat_alpha(r, dr, alpha_geom):
    if alpha_geom == 2:
        return r ** 2 + 1/12 * dr ** 2
    elif alpha_geom == 1:
        return r
    else:
        raise ValueError("Unknown geometry / not for cartesian coordinates")
    
@partial(jax.jit, static_argnames=['alpha_geom'])
def _center_of_volume(r, dr, alpha_geom):
    if alpha_geom == 1:
        return (r ** 2 + 1/12 * dr ** 2) / r ** 2 * r
    elif alpha_geom == 2:
        r_hat = _r_hat_alpha(r, dr, alpha_geom)
        return (r ** 2 + 1/4 * dr ** 2) / r_hat * r
    