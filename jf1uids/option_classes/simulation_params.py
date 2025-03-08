from typing import NamedTuple

from jf1uids._physics_modules._cosmic_rays.cosmic_ray_options import CosmicRayParams
from jf1uids._physics_modules._stellar_wind.stellar_wind_options import WindParams

class SimulationParams(NamedTuple):
    """Different from the simulation configuration, the simulation parameters
        do not require recompilation when changed. The simulation can be 
        differentiated with respect to them.
    """

    #: The Courant-Friedrichs-Lewy number, a factor
    #: in the time step calculation.
    C_cfl: float = 0.8

    #: Gravitational constant.
    gravitational_constant: float = 1.0

    #: The adiabatic index of the gas.
    gamma: float = 5/3

    #: The maximum time step.
    dt_max: float = 0.001

    #: The final time of the simulation.
    t_end: float = 0.2

    # parameters of physics modules

    #: The parameters of the stellar wind module.
    wind_params: WindParams = WindParams()

    #: Cosmic ray parameters
    cosmic_ray_params: CosmicRayParams = CosmicRayParams()