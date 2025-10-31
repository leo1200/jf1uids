from typing import Optional, Tuple
import random

from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids.initial_condition_generation.turb import create_turb_field

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
import astropy.constants as c
from corrector_src.utils.downaverage import downaverage_states, downaverage_state

import corrector_src.data.blast_creation as blast

from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    # CorrectorParams,
    # CorrectorConfig,
    CorrectorParams,
    CorrectorConfig,
)
from omegaconf import OmegaConf
import numpy as np
import jax.numpy as jnp
import jax.random
import time


class dataset:
    def __init__(self, scenarios_to_use: list[int], cfg_data: OmegaConf):
        self.num_scenarios = len(scenarios_to_use)
        self.complete_scenario_list = [
            "mhd_blast",
            "turbulence",
            "turbulence_mhd",
        ]
        self.scenario_list = [self.complete_scenario_list[i] for i in scenarios_to_use]
        self.default_resolution = cfg_data.hr_res
        self.default_downscaling_factor = cfg_data.downscaling_factor
        self.cfg_data = cfg_data

        # 🧭 Scenario dispatch map
        self._scenario_dispatch = {
            "mhd_blast": self._init_mhd_blast,
            "turbulence": self._init_turbulence,
            "turbulence_mhd": self._init_turbulence_mhd,
        }

        if self.cfg_data.use_specific_snapshot_timepoints and self.cfg_data.snapshot_timepoints is not None:
            self.cfg_data.num_snapshots = len(self.cfg_data.snapshot_timepoints)
            print(f"Returning snapshots with specific snapshots {self.cfg_data.snapshot_timepoints}")
        elif self.cfg_data.return_snapshots: 
            print(f"Returning {self.cfg_data.num_snapshots} snapshots")


    def _init_mhd_blast(self, resolution: int, **overrides):
        state_tuple = blast.randomized_initial_blast_state(
            resolution, cfg_data=self.cfg_data, rng_seed=overrides.get("rng_seed")
        )
        return self._apply_overrides_to_sim_tuple(state_tuple, **overrides)

    def _init_turbulence(self, resolution: int, **overrides):
        state_tuple = self.randomized_turbulent_initial_state(
            resolution, mhd=False, rng_seed=overrides.get("rng_seed")
        )
        return self._apply_overrides_to_sim_tuple(state_tuple, **overrides)

    def _init_turbulence_mhd(self, resolution: int, **overrides):
        state_tuple = self.randomized_turbulent_initial_state(
            resolution, mhd=True, rng_seed=overrides.get("rng_seed")
        )
        return self._apply_overrides_to_sim_tuple(state_tuple, **overrides)

    def _apply_overrides_to_sim_tuple(
        self,
        sim_tuple: Tuple[
            np.ndarray,
            SimulationConfig,
            SimulationParams,
            HelperData,
            RegisteredVariables,
            int,
        ],
        **overrides,
    ) -> Tuple[
        np.ndarray,
        SimulationConfig,
        SimulationParams,
        HelperData,
        RegisteredVariables,
        int,
    ]:
        """
        Takes a standard simulation tuple and applies optional overrides like
        corrector_config, corrector_params, or any future dynamic fields.
        """
        state, config, params, helper, reg_vars, seed = sim_tuple

        if (
            "corrector_config" in overrides
            and overrides["corrector_config"] is not None
        ):
            config = config._replace(corrector_config=overrides["corrector_config"])
        if (
            "corrector_params" in overrides
            and overrides["corrector_params"] is not None
        ):
            params = params._replace(corrector_params=overrides["corrector_params"])

        return state, config, params, helper, reg_vars, seed

    def sim_initializator(
        self,
        resolution: Optional[int] = None,
        scenario: Optional[str | int] = None,
        rng_seed: Optional[str | int | float] = None,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
        **overrides,
    ) -> Tuple[
        np.ndarray,
        SimulationConfig,
        SimulationParams,
        HelperData,
        RegisteredVariables,
        int,
    ]:
        """returns initial_state, config, params, helper_data, registered_variables, seed"""

        if isinstance(scenario, str) and scenario not in self.scenario_list:
            raise NameError(f"scenario should be in list {self.scenario_list}")
        if scenario is None:
            scenario = random.choice(self.scenario_list)

        resolution = resolution or self.default_resolution

        # Update overrides
        overrides.update({
            k: v for k, v in {
                "corrector_config": corrector_config,
                "corrector_params": corrector_params,
                "rng_seed": rng_seed,
            }.items() if v is not None
        })

        # Dispatch dynamically
        if scenario not in self._scenario_dispatch:
            raise ValueError(f"Unknown scenario: {scenario}")

        return self._scenario_dispatch[scenario](resolution=resolution, **overrides)

    def initialize_integrate(
        self,
        resolution: Optional[int] = None,
        scenario: Optional[str | int] = None,
        rng_seed: Optional[int] = None,
        downscale_factor: Optional[int] = None,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
    )-> Tuple[np.ndarray, Tuple[np.ndarray, SimulationConfig, SimulationParams, HelperData, RegisteredVariables, int]]:

        """
        integrates a simulation and returns the states with the config, resolution, and seed given"""
        initial_state, config, params, helper_data, registered_variables, rng_seed = (
            self.sim_initializator(
                resolution,
                scenario,
                rng_seed,
                corrector_config,
                corrector_params,
            )
        )

        config = finalize_config(config, initial_state.shape)
        final_states = time_integration(
            initial_state, config, params, helper_data, registered_variables
        )
        states = final_states.states

        if downscale_factor is not None:
            states = downaverage_states(states, downscale_factor)

        return states, (
            initial_state,
            config,
            params,
            helper_data,
            registered_variables,
            rng_seed,
        )

    def hr_lr_initializator(
        self,
        resolution: Optional[int] = None,
        downscale: Optional[int] = None,
        scenario: Optional[str | int] = None,
        rng_seed: Optional[str | int | float] = None,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
        **overrides,
    ) -> Tuple[
        Tuple[
            np.ndarray,
            SimulationConfig,
            SimulationParams,
            HelperData,
            RegisteredVariables,
        ],
        Tuple[
            np.ndarray,
            SimulationConfig,
            SimulationParams,
            HelperData,
            RegisteredVariables,
        ],
    ]:
        """returns hr-lr pair of (state, config, params, helper_data, registered_vars)"""

        if isinstance(scenario, str) and scenario not in self.scenario_list:
            raise NameError(f"scenario should be in list {self.scenario_list}")
        if scenario is None:
            scenario = random.choice(self.scenario_list)

        resolution = resolution or self.default_resolution
        downscale = downscale or self.default_downscaling_factor

        # Include RNG override only (CNN parameters are for LR only)
        if rng_seed is not None:
            overrides["rng_seed"] = rng_seed

        # Dispatch dynamically to get HR state
        if scenario not in self._scenario_dispatch:
            raise ValueError(f"Unknown scenario: {scenario}")

        (
            initial_state_hr,
            config_hr,
            params_hr,
            helper_data_hr,
            registered_variables_hr,
            rng_seed,
        ) = self._scenario_dispatch[scenario](resolution=resolution, **overrides)

        # Downscale for LR state
        initial_state_lr = downaverage_state(
            state=initial_state_hr, downsample_factor=downscale
        )
        config_lr = finalize_config(
            config_hr._replace(num_cells=resolution // downscale),
            initial_state_lr.shape,
        )
        helper_data_lr = get_helper_data(config_lr)
        registered_variables_lr = get_registered_variables(config_lr)
        params_lr = params_hr  # params independent of config

        # Apply CNN overrides ONLY to LR state
        if corrector_config is not None:
            config_lr = config_lr._replace(corrector_config=corrector_config)
        if corrector_params is not None:
            params_lr = params_lr._replace(corrector_params=corrector_params)

        return (
            (
                initial_state_hr,
                config_hr,
                params_hr,
                helper_data_hr,
                registered_variables_hr,
            ),
            (
                initial_state_lr,
                config_lr,
                params_lr,
                helper_data_lr,
                registered_variables_lr,
            ),
        )

    def train_initializator(
        self,
        resolution: Optional[int] = None,
        downscale: Optional[int] = None,
        scenario: Optional[str | int] = None,
        rng_seed: Optional[str | int | float] = None,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
        **overrides,
    ) -> Tuple[
        np.ndarray,
        Tuple[
            np.ndarray,
            SimulationConfig,
            SimulationParams,
            HelperData,
            RegisteredVariables,
        ],
    ]:
        """function tailored to use while training the corrector_config
        returns hr states (DOWNSCALED if downscale is given) and lr  state, config, params, helper_data, registered_vars
        *if the corrector is given it will only be applied to the lr configuration and parameters"""

        if isinstance(scenario, str) and scenario not in self.scenario_list:
            raise NameError(f"scenario should be in list {self.scenario_list}")
        if scenario is None:
            scenario = random.choice(self.scenario_list)

        resolution = resolution or self.default_resolution
        downscale = downscale or self.default_downscaling_factor
        # Include RNG override only (CNN parameters are for LR only)
        if rng_seed is not None:
            overrides["rng_seed"] = rng_seed

        # Dispatch dynamically to get HR state
        if scenario not in self._scenario_dispatch:
            raise ValueError(f"Unknown scenario: {scenario}")

        (
            hr_states,
            (
                initial_state_hr,
                config_hr,
                params_hr,
                helper_data_hr,
                registered_variables_hr,
                rng_seed,
            ),
        ) = self.initialize_integrate(
            resolution=resolution,
            scenario=scenario,
            rng_seed=rng_seed,
            downscale_factor=downscale,
        )
        initial_state_lr = hr_states[0]
        config_lr = finalize_config(
            config_hr._replace(num_cells=resolution // downscale),
            initial_state_lr.shape,
        )
        helper_data_lr = get_helper_data(config_lr)
        registered_variables_lr = get_registered_variables(config_lr)
        params_lr = params_hr  # params independent of config

        # Apply CNN overrides ONLY to LR state
        if corrector_config is not None:
            config_lr = config_lr._replace(corrector_config=corrector_config)
        if corrector_params is not None:
            params_lr = params_lr._replace(corrector_params=corrector_params)

        return (
            hr_states,
            (
                initial_state_lr,
                config_lr,
                params_lr,
                helper_data_lr,
                registered_variables_lr,
            ),
        )

    def randomized_turbulent_initial_state(
        self,
        num_cells: int,
        mhd: bool,
        rng_seed: Optional[int] = None,
    ) -> Tuple[
        np.ndarray,
        SimulationConfig,
        SimulationParams,
        HelperData,
        RegisteredVariables,
        int,
    ]:
        "Creates a turbulent initial state with mhd"
        adiabatic_index = 5 / 3
        box_size = 1.0
        dt_max = 0.1

        # setup simulation config
        config = SimulationConfig(
            runtime_debugging=self.cfg_data.debug,
            first_order_fallback=False,
            progress_bar=False,
            dimensionality=3,
            num_ghost_cells=2,
            box_size=box_size,
            num_cells=num_cells,
            mhd=mhd,
            fixed_timestep=self.cfg_data.fixed_timestep,
            differentiation_mode=BACKWARDS,
            riemann_solver=HLL,
            limiter=0,
            return_snapshots=self.cfg_data.return_snapshots,
            num_snapshots=self.cfg_data.num_snapshots,
            use_specific_snapshot_timepoints=self.cfg_data.use_specific_snapshot_timepoints,
            boundary_settings=BoundarySettings(),
            num_checkpoints=self.cfg_data.num_checkpoints,
            # boundary_settings=BoundarySettings(
            #    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            #    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            #    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            # ),
        )

        helper_data = get_helper_data(config)
        registered_variables = get_registered_variables(config)

        # setup the unit system
        code_length = 3 * u.parsec
        code_mass = 1 * u.M_sun
        code_velocity = 100 * u.km / u.s
        code_units = CodeUnits(code_length, code_mass, code_velocity)

        # time domain
        C_CFL = 0.4  # Courant-Friedrichs-Lewy number
        t_final = 1.0 * 1e4 * u.yr
        # t_end = t_final.to(code_units.code_time).value
        t_end = self.cfg_data.t_end
        # turbulence
        wanted_rms = 50 * u.km / u.s
        dt_max = 0.1

        # set the simulation parameters
        params = SimulationParams(
            C_cfl=C_CFL,
            dt_max=dt_max,
            gamma=adiabatic_index,
            t_end=t_end,
            snapshot_timepoints=jnp.array(self.cfg_data.snapshot_timepoints),

        )

        # homogeneous initial state
        rho_0 = 2 * c.m_p / u.cm**3
        p_0 = 3e4 * u.K / u.cm**3 * c.k_B

        density = (
            jnp.ones((num_cells, num_cells, num_cells))
            * rho_0.to(code_units.code_density).value
        )

        # turbulence parameters
        turbulence_slope = -2
        kmin = 2
        kmax = 64

        p = (
            jnp.ones((num_cells, num_cells, num_cells))
            * p_0.to(code_units.code_pressure).value
        )
        if rng_seed is None:
            rng_seed = int(time.time() * 1e6) % (2**32 - 1)
        key = jax.random.key(rng_seed)

        keys = jax.random.split(key, 3)
        u_x = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[0]
        )
        u_y = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[1]
        )
        u_z = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[2]
        )

        # scale the turbulence to the desired rms velocity
        rms_vel = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))

        u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_z = u_z / rms_vel * wanted_rms.to(code_units.code_velocity).value
        # construct primitive state

        if mhd:
            grid_spacing = config.box_size / config.num_cells
            x = jnp.linspace(
                grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
            )
            y = jnp.linspace(
                grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
            )
            z = jnp.linspace(
                grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
            )

            X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

            B_0 = 1 / np.sqrt(2)
            B_x = jnp.zeros_like(X)
            B_y = jnp.zeros_like(X)
            B_z = B_0 * jnp.ones_like(X)

            initial_state = construct_primitive_state(
                config=config,
                registered_variables=registered_variables,
                density=density,
                velocity_x=u_x,
                velocity_y=u_y,
                velocity_z=u_z,
                gas_pressure=p,
                magnetic_field_x=B_x,
                magnetic_field_y=B_y,
                magnetic_field_z=B_z,
            )
        else:
            initial_state = construct_primitive_state(
                config=config,
                registered_variables=registered_variables,
                density=density,
                velocity_x=u_x,
                velocity_y=u_y,
                velocity_z=u_z,
                gas_pressure=p,
            )

        return (
            initial_state,
            config,
            params,
            helper_data,
            registered_variables,
            rng_seed,
        )
