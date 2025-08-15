from typing import NamedTuple
from jax import numpy as jnp
import jax

class BinaryConfig(NamedTuple):
    binary: bool = False
    deposit_particles: str = "ngp"  # Options: "ngp", "cic", "tsc"

class BinaryParams(NamedTuple):
    masses: jnp.ndarray = None 
    binary_state: jnp.ndarray = None

    
