from typing import NamedTuple
import jax.numpy as jnp

# wind injection schemes
MEO = 0 # momentum and energy overwrite
EI = 1 # thermal energy injection
MEI = 2 # momentum and energy injection

class WindConfig(NamedTuple):
    stellar_wind: bool = False
    num_injection_cells: int = 10
    wind_injection_scheme: int = EI
    trace_wind_density: bool = False
    real_wind_params: bool = False

class WindParams(NamedTuple):
    wind_mass_loss_rates: jnp.array = None
    wind_final_velocities: jnp.array = None
    wind_injection_positions: jnp.array = jnp.array([[0.0, 0.0, 0.0]])
    real_params: jnp.array = None
    # only necesarry for the MEO injection scheme
    pressure_floor: float = 100000.0