import jax.numpy as jnp
import jax

@jax.jit
def minmod(a, b):
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))

@jax.jit
def meanmod(a, b):
    return 0.5 * (a + b)

@jax.jit
def maxmod(a, b):
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.maximum(jnp.abs(a), jnp.abs(b))

@jax.jit
def superbee(a, b):
    res = minmod(maxmod(a, b), minmod(2 * a, 2 * b))
    res = res * (a * b <= 0)
    return res