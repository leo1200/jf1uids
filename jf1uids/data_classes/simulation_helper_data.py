from typing import NamedTuple
import jax.numpy as jnp
from jf1uids.geometry.geometry import _center_of_volume, _r_hat_alpha

# Helper data like the radii and cell volumes 
# in the simulation or cooling tables etc.

class HelperData(NamedTuple):
    geometric_centers: jnp.ndarray = None
    volumetric_centers: jnp.ndarray = None
    r_hat_alpha: jnp.ndarray = None

def get_helper_data(config):
    dx = config.box_size / (config.num_cells - 1)
    if config.alpha_geom == 0:
        r = jnp.linspace(0, config.box_size, config.num_cells)
        r_hat = dx * jnp.ones_like(r) # not really
        return HelperData(geometric_centers = r, r_hat_alpha = r_hat)
    elif config.alpha_geom == 2 or config.alpha_geom == 1:
        # r = jnp.linspace(- 3 * dx/2, config.box_size - 3 * dx / 2, config.num_cells)
        r = jnp.linspace(dx / 2, config.box_size + dx / 2, config.num_cells)
        volumetric_centers = _center_of_volume(r, dx, config.alpha_geom)
        r_hat = _r_hat_alpha(r, dx, config.alpha_geom)
        return HelperData(geometric_centers = r, volumetric_centers = volumetric_centers, r_hat_alpha = r_hat)