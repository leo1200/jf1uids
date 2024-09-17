import jax.numpy as jnp
import jax

# TODO: rewrite limiters to the one-argument convention

@jax.jit
def _minmod(a, b):
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))