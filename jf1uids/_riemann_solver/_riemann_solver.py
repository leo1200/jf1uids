# general imports
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# general jf1uids
from jf1uids._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from jf1uids._riemann_solver.hll import _hll_solver, _hllc_solver
from jf1uids.fluid_equations.registered_variables import RegisteredVariables

# fluid stuff
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, speed_of_sound
from jf1uids.fluid_equations.euler import _euler_flux
from jf1uids.option_classes.simulation_config import HLL, HLLC, HLLC_LM, STATE_TYPE, SimulationConfig

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'registered_variables', 'flux_direction_index'])
def _riemann_solver(primitives_left: STATE_TYPE, primitives_right: STATE_TYPE, gamma: Union[float, Float[Array, ""]], config: SimulationConfig, registered_variables: RegisteredVariables, flux_direction_index: int) -> STATE_TYPE:
    """Wrapper function for the Riemann solver."""
    if config.riemann_solver == HLL:
        return _hll_solver(primitives_left, primitives_right, gamma, config, registered_variables, flux_direction_index)
    elif config.riemann_solver == HLLC:
        return _hllc_solver(primitives_left, primitives_right, gamma, config, registered_variables, flux_direction_index)
    else:
        raise ValueError("Riemann solver not supported.")