import jax.numpy as jnp

def minmod(a, b):
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))

def maxmod(a, b):
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.maximum(jnp.abs(a), jnp.abs(b))

def superbee(a, b):
    res = minmod(maxmod(a, b), minmod(2 * a, 2 * b))
    res = res.at[a * b <= 0].set(0)
    return res