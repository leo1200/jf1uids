import jax
from functools import partial

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids._physics_modules._stellar_wind.stellar_wind import _wind_injection

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def _run_physics_modules(primitive_state: Float[Array, "num_vars num_cells"], dt: Float[Array, ""], config: SimulationConfig, params: SimulationParams, helper_data: HelperData) -> Float[Array, "num_vars num_cells"]:
    """Run all the physics modules. The physics modules are switched on/off and
    configured in the simulation configuration. Parameters for the physics modules
    (with respect to which the simulation can be differentiated) are stored in the
    simulation parameters.

    Args:
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        The primitive state array with the physics modules applied.
    """
    
    # stellar wind
    if config.wind_config.stellar_wind:
        primitive_state = _wind_injection(primitive_state, dt, config, params, helper_data)

    primitive_state = _boundary_handler(primitive_state, config.left_boundary, config.right_boundary)

    return primitive_state
