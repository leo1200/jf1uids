from typing import NamedTuple

class CosmicRayConfig(NamedTuple):
    cosmic_rays: bool = False
    diffusive_shock_acceleration: bool = False

class CosmicRayParams(NamedTuple):
    diffusive_shock_acceleration_start_time: float = 0.0
    diffusive_shock_acceleration_efficiency: float = 0.1