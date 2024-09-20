import jax
from functools import partial

from jf1uids.geometry.boundaries import _boundary_handler
from jf1uids.physics_modules.stellar_wind.stellar_wind import _wind_injection

@partial(jax.jit, static_argnames=['config'])
def run_physics_modules(primitive_state, dt, config, params, helper_data):
    
    # stellar wind
    if config.wind_config.stellar_wind:
        primitive_state = _wind_injection(primitive_state, dt, config, params, helper_data)

    primitive_state = _boundary_handler(primitive_state, config.left_boundary, config.right_boundary)

    return primitive_state
