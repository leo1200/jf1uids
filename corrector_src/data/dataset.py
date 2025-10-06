from typing import Optional, Tuple
import random

from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import time_integration
from corrector_src.utils.downaverage import downaverage_states

import corrector_src.data.blast_creation as blast
import corrector_src.data.turbulence_creation as turb
from omegaconf import OmegaConf
import numpy as np
from jf1uids import SimulationConfig, SimulationParams
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables


class dataset:
    def __init__(
        self, scenarios_to_use: list[int], randomize: bool, cfg_data: OmegaConf
    ):
        self.num_scenarios = len(scenarios_to_use)
        self.complete_scenario_list = [
            "mhd_blast",
            "turbulence",
            "turbulence_mhd",
        ]
        self.scenario_list = self.complete_scenario_list[scenarios_to_use]
        self.randomize = randomize
        self.default_resolution = cfg_data.hr_res
        self.cfg_data = cfg_data

    def return_initializator(
        self, resolution: Optional[int] = None, scenario: Optional[str | int] = None
    ) -> Tuple[
        np.ndarray,
        SimulationConfig,
        SimulationParams,
        HelperData,
        RegisteredVariables,
        int,
    ]:
        "returns initial_state, config, params, helper_data, registered_variables seed"
        # needs randomizer returnal!!!
        if isinstance(scenario, str) and scenario not in self.scenario_list:
            raise NameError(f"scenario should be in list {self.scenario_list}")
        if scenario is None:
            scenario = self.scenario_list[random.randint(0, self.num_scenarios - 1)]
        if resolution is None:
            resolution = self.default_resolution
        match scenario:
            case "mhd_blast":
                return blast.randomized_initial_blast_state(
                    resolution, cfg_data=self.cfg_data
                )
            case "turbulence":
                return turb.randomized_turbulent_initial_state(
                    resolution, self.cfg_data, mhd=False
                )
            case "turbulence_mhd":
                return turb.randomized_turbulent_initial_state(
                    resolution, self.cfg_data, mhd=True
                )

    def initialize_integrate(
        self,
        resolution: Optional[int] = None,
        scenario: Optional[str | int] = None,
        downscale_factor: Optional[int] = False,
    ) -> np.ndarray:
        initial_state, config, params, helper_data, registered_variables = (
            self.return_initializator(resolution, scenario)
        )

        config = finalize_config(config, initial_state.shape)
        final_states = time_integration(
            initial_state, config, params, helper_data, registered_variables
        )
        states = final_states.states

        if downscale_factor is not None:
            states = downaverage_states(states, downscale_factor)

        return states
