from typing import NamedTuple
import jax.numpy as jnp

# Helper data like the radii and cell volumes 
# in the simulation or cooling tables etc.

class SimulationHelperData(NamedTuple):
    geometric_centers: jnp.ndarray = None
    volumetric_centers: jnp.ndarray = None
    r_hat_alpha: jnp.ndarray = None