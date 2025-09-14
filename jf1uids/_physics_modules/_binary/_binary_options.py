from typing import NamedTuple
from jax import numpy as jnp

# mass deposit schemes
NGP = 0 #nearest grid point
CIC = 1 #cloud in cell
TSC = 2 #triangular shaped cloud

class BinaryConfig(NamedTuple):
    binary: bool = False
    deposit_particles: int = NGP 
    central_object_only: bool = False

class BinaryParams(NamedTuple):
    masses: jnp.ndarray = None 
    binary_state: jnp.ndarray = None
    