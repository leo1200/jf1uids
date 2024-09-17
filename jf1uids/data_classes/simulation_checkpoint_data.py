from typing import NamedTuple
import jax.numpy as jnp

class CheckpointData(NamedTuple):
    # Checkpoint data
    times: jnp.ndarray = None
    total_mass_proxy: jnp.ndarray = None
    total_energy_proxy: jnp.ndarray = None
    states: jnp.ndarray = None
    current_checkpoint: int = 0
    num_iterations: int = 0

    runtime: float = 0.0