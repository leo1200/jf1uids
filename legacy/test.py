import jax
import jax.numpy as jnp

# for now using CPU as of outdated NVIDIA Driver
jax.config.update('jax_platform_name', 'cpu')

@jax.jit
def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(5.0)
print(selu(x))