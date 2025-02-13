from typing import NamedTuple
import jax.numpy as jnp

class SnapshotData(NamedTuple):
    """Return format for the time integration, when snapshots are requested."""

    #: The times at which the snapshots were taken.
    time_points: jnp.ndarray = None

    #: The primitive states at the times the snapshots were taken.
    states: jnp.ndarray = None

    #: The total mass at the times the snapshots were taken.
    total_mass: jnp.ndarray = None

    #: The total energy at the times the snapshots were taken.
    total_energy: jnp.ndarray = None

    #: internal energy
    internal_energy: jnp.ndarray = None

    #: kinetic energy
    kinetic_energy: jnp.ndarray = None

    #: gravitational energy
    gravitational_energy: jnp.ndarray = None

    # The runtime of the simulation-loop.
    runtime: float = 0.0

    #: Number of timesteps taken.
    num_iterations: int = 0

    #: The current checkpoint, used internally.
    current_checkpoint: int = 0



