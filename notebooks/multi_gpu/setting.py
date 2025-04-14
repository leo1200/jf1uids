from functools import partial
import jax
import jax.numpy as jnp

#  set XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

# timeit
import timeit

@partial(jax.jit, static_argnames=['axis', 'set_index', 'get_index', 'var_index', 'factor'])
def _set_specific_var_along_axis(
    primitive_state: jnp.ndarray,
    axis: int,
    set_index: int,
    get_index: int,
    var_index: int,
    factor: float
) -> jnp.ndarray:

    s_set = (var_index,) + (slice(None),) * (axis - 1) + (set_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)
    s_get = (var_index,) + (slice(None),) * (axis - 1) + (get_index,) + (slice(None),)*(primitive_state.ndim - axis - 1)

    primitive_state = primitive_state.at[s_set].set(factor * primitive_state[s_get])

    return primitive_state

# get random example array of sice 512^3
arr = jax.random.normal(jax.random.PRNGKey(0), (5, 512, 512, 512))

set_index = 0
get_index = -1
axis = 1


new_arr = _set_specific_var_along_axis(
    arr,
    axis=axis,
    set_index=set_index,
    get_index=get_index,
    var_index=0,
    factor=1.0
).block_until_ready()

# measure execution time
execution_time = timeit.repeat(
    lambda: _set_specific_var_along_axis(
        arr,
        axis=axis,
        set_index=set_index,
        get_index=get_index,
        var_index=0,
        factor=1.0
    ),
    number=10,
    repeat=10,
)
print(min(execution_time) / 10 * 1e6)  # in microseconds