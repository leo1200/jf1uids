import jax
from functools import partial

from jf1uids.geometry.boundaries import _boundary_handler
from jf1uids.physics_modules.stellar_wind.stellar_wind import wind_source


@partial(jax.jit, static_argnames=['config'])
def add_physical_sources(primitive_state, dt, config, params, helper_data):
    
    # stellar wind
    if config.stellar_wind:
        source_term = wind_source(params.wind_params, primitive_state, helper_data.volumetric_centers, config.dx)
        primitive_state = primitive_state.at[:, :].add(dt * source_term)

    primitive_state = _boundary_handler(primitive_state, config.left_boundary, config.right_boundary)

    return primitive_state
