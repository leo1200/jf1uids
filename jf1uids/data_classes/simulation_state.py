from flax import struct

import jax.numpy as jnp

from types import NoneType
from typing import NamedTuple, Union

from jf1uids.option_classes.simulation_config import STATE_TYPE

@struct.dataclass
class SimulationState:
    """
    The simulation state.
    """

    #: The gas state.
    gas_state: Union[STATE_TYPE, NoneType] = None

    #: Magnetic field state.
    magnetic_field_state: Union[jnp.ndarray, NoneType] = None