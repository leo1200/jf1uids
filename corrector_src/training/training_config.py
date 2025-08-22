# numerics
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# fluids
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config

# jf1uids constants
from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    FORWARDS,
    HLL,
    HLLC,
    MINMOD,
    OSHER,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

# units
from jf1uids import CodeUnits
from astropy import units as u
from typing import NamedTuple, Callable, Optional, Dict

"""
create a simulation config tailored to our training, this is:
    1. Fixed timested 
"""    
def create_config(num_cells, num_snapshots = 80, ):
    adiabatic_index = 5 / 3
    box_size = 1.0
    fixed_timestep = False
    dt_max = 0.1
    mhd = True

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging=False,
        first_order_fallback=False,
        progress_bar=True,
        dimensionality=3,
        num_ghost_cells=2,
        box_size=box_size,
        num_cells=num_cells,
        mhd=mhd,
        fixed_timestep=fixed_timestep,
        differentiation_mode=FORWARDS,
        riemann_solver=HLL,
        limiter=0,
        return_snapshots=True,
        num_snapshots=80,
        # boundary_settings=BoundarySettings(
        #    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        #    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        #    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        # ),
    )
    return config

class TrainingConfig(NamedTuple):
    # Intermediate loss computation settings
    compute_intermediate_losses: bool = True
    n_look_behind: int = 10
    
    # Loss function and related settings
    loss_function: Optional[Callable] = None
    loss_weights: Optional[Dict[str, float]] = None
    use_relative_error: bool = False
    
    # Ground truth data (if using supervised learning)
    ground_truth_snapshots: Optional[jnp.ndarray] = None
    
    # Training state tracking
    accumulated_loss: float = 0.0
    loss_count: int = 0
    
    # Optional spatial mask for loss computation
    loss_mask: Optional[jnp.ndarray] = None
    
    # Downscaling method for ground truth
    downscale_method: str = 'average'

    current_checkpoint_total: int = 0
    current_checkpoint_chunk: int = 0

    return_full_sim: bool = True
