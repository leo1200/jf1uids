import jax.numpy as jnp
import jax

# TODO: rewrite limiters to the one-argument convention

@jax.jit
def _minmod(a, b):
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))

def _minmod3(a, b, c):
    """Minmod function for three arguments in JAX."""
    same_sign = (jnp.sign(a) == jnp.sign(b)) & (jnp.sign(b) == jnp.sign(c))
    return jnp.where(same_sign, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c))), 0.0)

def _double_minmod(a, b):
    """
    Double minmod limiter.
    """
    return jnp.where(
        a * b > 0, 
        _minmod3((a+b)/2, 2*a, 2*b),
        0.0
    )

    
@jax.jit
def _maxmod(a, b):
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.maximum(jnp.abs(a), jnp.abs(b))

def _superbee(a, b):
    """
    Superbee limiter.
    """
    return jnp.where(a * b > 0, _minmod(
        _maxmod(a, b),
        _minmod(2 * a, 2 * b)
    ), 0.0)

# @jax.jit
# def _minmod(r):
#     return jnp.maximum(0, jnp.minimum(1,r))