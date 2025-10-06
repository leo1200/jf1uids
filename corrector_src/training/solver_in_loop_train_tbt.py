import jax
import jax.numpy as jnp

# type checking
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from typing import Union, Callable, Tuple

# jf1uids constants
from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    CARTESIAN,
    FORWARDS,
    STATE_TYPE,
)

# jf1uids containers
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData

# jf1uids functions
from jf1uids._state_evolution.evolve_state import _evolve_state
from jf1uids._physics_modules.run_physics_modules import _run_physics_modules
from jf1uids.time_stepping._timestep_estimator import (
    _cfl_time_step,
    _source_term_aware_time_step,
)
from jf1uids.fluid_equations.total_quantities import (
    calculate_internal_energy,
    calculate_total_mass,
)
from jf1uids.fluid_equations.total_quantities import (
    calculate_total_energy,
    calculate_kinetic_energy,
    calculate_gravitational_energy,
)

from corrector_src.training.training_config import TrainingConfig
from corrector_src.utils.downaverage import downaverage_states, downaverage_state

import optax
import equinox as eqx

from jax.experimental import checkify

from functools import partial


def time_integration_training_tbt( #timestep by timestep (poco a poco lava la vieja el coco)
    initial_state_hr: STATE_TYPE,
    config_hr: SimulationConfig,
    config_lr: SimulationConfig,
    params: SimulationParams,
    helper_data_hr: HelperData,
    helper_data_lr: HelperData,
    registered_variables: RegisteredVariables,
    training_config: TrainingConfig,
    optimizer: optax.GradientTransformation,
    loss_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    opt_state: optax.OptState,
    save_full_sim: bool = False,
) -> Tuple[jnp.ndarray, eqx.Module, optax.OptState, SnapshotData]:
    assert config_hr.num_checkpoints == config_lr.num_checkpoints, "num_checkpoints different from hr to lr"
    assert config_hr.runtime_debugging == config_lr.runtime_debugging, "runtime_debugging different from hr to lr"
    assert config_hr.dimensionality == config_lr.dimensionality, "dimensionality different from hr to lr"
    assert config_hr.mhd == config_lr.mhd, "mhd different from hr to lr"
    assert config_hr.self_gravity == config_lr.self_gravity, "self_gravity different from hr to lr"
    assert config_hr.box_size == config_lr.box_size, "box_size different from hr to lr"
    assert config_hr.riemann_solver == config_lr.riemann_solver, "riemann_solver different from hr to lr"
    assert config_hr.num_ghost_cells == config_lr.num_ghost_cells, "num_ghost_cells different from hr to lr"
    assert config_hr.boundary_settings == config_lr.boundary_settings, "boundary_settings different from hr to lr"
    assert config_hr.fixed_timestep == config_lr.fixed_timestep, "fixed_timestep different from hr to lr"
    assert config_hr.exact_end_time == config_lr.exact_end_time, "exact_end_time different from hr to lr"
    assert config_hr.num_timesteps == config_lr.num_timesteps, "num_timesteps different from hr to lr"
    assert config_hr.differentiation_mode == config_lr.differentiation_mode, "differentiation_mode different from hr to lr"
    assert config_hr.return_snapshots == config_lr.return_snapshots, "return_snapshots different from hr to lr"
    assert config_hr.num_cells // training_config.downscale_factor == config_lr.num_cells, "downscaled cells different from lr cells"
    assert config_hr.geometry == config_lr.geometry, "geometry not the sames"
    if config_hr.runtime_debugging:
        checked_integration = checkify.checkify(_time_integration_training)

        err, final_state = checked_integration(
            initial_state_hr,
            config_hr,
            config_lr,
            params,
            helper_data_hr,
            helper_data_lr,
            registered_variables,
            training_config,
            optimizer,
            loss_function,
            opt_state,
            save_full_sim,
        )

        return err, final_state

    else:
        return _time_integration_training(
            initial_state_hr,
            config_hr,
            config_lr,
            params,
            helper_data_hr,
            helper_data_lr,
            registered_variables,
            training_config,
            optimizer,
            loss_function,
            opt_state,
            save_full_sim,
        )
@partial(
    jax.jit,
    static_argnames=[
        "config_lr",
        "config_hr",
        "registered_variables",
        "training_config",
        "optimizer",
        "save_full_sim",
        "loss_function",
    ],
)
def _time_integration_training(
    initial_state_hr: STATE_TYPE,
    config_hr: SimulationConfig,
    config_lr: SimulationConfig,
    params: SimulationParams,
    helper_data_hr: HelperData,
    helper_data_lr: HelperData,
    registered_variables: RegisteredVariables,
    training_config: TrainingConfig,
    optimizer: optax.GradientTransformation,
    loss_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    opt_state: optax.OptState,
    save_full_sim: bool = False,
) -> Tuple[jnp.ndarray, eqx.Module, optax.OptState, SnapshotData]:
    """
    Runs training where the high-resolution simulation advances without the neural corrector
    to produce 'ground-truth' lag states. The low-resolution simulation (also without
    the corrector) advances and the loss is computed between LR-evolved states and a
    downsampled HR lag history.

    Returns:
        losses: Array of losses per batch (num_batches)
        final_network_params: final network params (unchanged if not training a network)
        final_opt_state: optimizer state
        final_sim_data_hr: stored HR snapshots (depending on save_full_sim)
    """
    total_steps = config_hr.num_timesteps
    data_lag = training_config.n_look_behind
    num_batches = total_steps // data_lag
    ghost_cells = config_hr.num_ghost_cells
    # helpers for padding/unpadding states with ghost cells
    def pad_state(state):
        if config_hr.geometry == CARTESIAN:
            original_shape = state.shape
            if config_hr.dimensionality == 1:
                state = jnp.pad(state, ((0, 0), (ghost_cells, ghost_cells)), mode="edge")
            elif config_hr.dimensionality == 2:
                state = jnp.pad(state, ((0, 0), (ghost_cells, ghost_cells), (ghost_cells, ghost_cells)), mode="edge")
            elif config_hr.dimensionality == 3:
                state = jnp.pad(
                    state, ((0, 0), (ghost_cells, ghost_cells), (ghost_cells, ghost_cells), (ghost_cells, ghost_cells)), mode="edge"
                )
            return state, original_shape
        else:
            return state, state.shape

    def unpad_state(state):
        if config_hr.geometry == CARTESIAN:
            if config_hr.dimensionality == 1:
                return jax.lax.slice_in_dim(state, ghost_cells, state.shape[1] - ghost_cells, axis=1)
            elif config_hr.dimensionality == 2:
                s = jax.lax.slice_in_dim(state, ghost_cells, state.shape[1] - ghost_cells, axis=1)
                return jax.lax.slice_in_dim(s, ghost_cells, s.shape[2] - ghost_cells, axis=2)
            elif config_hr.dimensionality == 3:
                s = jax.lax.slice_in_dim(state, ghost_cells, state.shape[1] - ghost_cells, axis=1)
                s = jax.lax.slice_in_dim(s, ghost_cells, s.shape[2] - ghost_cells, axis=2)
                return jax.lax.slice_in_dim(s, ghost_cells, s.shape[3] - ghost_cells, axis=3)
        return state


    # Prepare padded initial states and storage
    initial_state_lr = downaverage_state(initial_state_hr, training_config.downscale_factor)
    initial_state_hr, original_shape_hr = pad_state(initial_state_hr)
    initial_state_lr, original_shape_lr = pad_state(initial_state_lr)

    # network params (kept for signature compatibility, might be unused if corrector disabled)
    neural_net_params = params.cnn_mhd_corrector_params.network_params

    # simulation single-step function used by both HR and LR scans
    def single_simulation_step(state, current_time, cfg, prm, help_data):
        # compute dt
        if not cfg.fixed_timestep:
            if cfg.source_term_aware_timestep:
                dt = jax.lax.stop_gradient(
                    _source_term_aware_time_step(
                        state, cfg, prm, help_data, registered_variables, current_time
                    )
                )
            else:
                dt = jax.lax.stop_gradient(
                    _cfl_time_step(
                        state,
                        cfg.grid_spacing,
                        prm.dt_max,
                        prm.gamma,
                        cfg,
                        registered_variables,
                        params.C_cfl,
                    )
                )
        else:
            dt = jnp.asarray(prm.t_end / cfg.num_timesteps)

        # run physics modules and evolution
        state = _run_physics_modules(
            state, dt, cfg, prm, help_data, registered_variables, current_time + dt
        )
        state = _evolve_state(
            state,
            dt,
            prm.gamma,
            prm.gravitational_constant,
            cfg,
            help_data,
            registered_variables,
        )
        current_time = current_time + dt
        return state, current_time

    # helper to run 'data_lag' steps and return (final_state, final_time, lag_buffer, sim_data_updated)
    def run_for_lag_steps(state_init, time_init, cfg, prm, save_full_sim_flag, help_data):
        lag_buffer = jnp.zeros((data_lag, *unpad_state(state_init).shape))
        # carry for scan: (state, current_time, sim_data, lag_buffer, step_idx)
        def body(carry, idx):
            state, current_time, lag_buf = carry

            # update sim_data if requested
            """
            sim_data_local = jax.lax.cond(
                save_full_sim_flag,
                lambda a, b, c, d, e: update_simulation_data(a, b, c, d, e),
                lambda a, b, c, d, e: c,
                current_time,
                state,
                sim_data_local,
                cfg,
                help_data
            )
            """
            # step
            state_new, time_new = single_simulation_step(state, current_time, cfg, prm, help_data)
            
            # put unpadded into lag buffer
            lag_buf = lag_buf.at[idx].set(unpad_state(state_new))

            return (state_new, time_new, lag_buf), None

        init_carry = (state_init, time_init, lag_buffer)
        (final_state, final_time, final_lag_buf), _ = jax.lax.scan(
            body, init_carry, jnp.arange(data_lag)
        )
        return final_state, final_time, final_lag_buf


    # initial carry for the outer batch scan
    batch_init_carry = (
        initial_state_hr,  # state_hr (padded)
        initial_state_lr,  # state_lr (padded)
        0.0,  # initial time
        neural_net_params,
        opt_state,
    )

    def batch_step(carry, batch_i):
        (
            state_hr,
            state_lr,
            current_time,
            network_params,
            opt_state_local,
        ) = carry

        # pack params for HR run (network irrelevant since cnn disabled)
        hr_params = params

        state_hr_after, time_hr_after, lag_hr = run_for_lag_steps(
            state_hr, current_time, config_hr, hr_params, save_full_sim, helper_data_hr
        )

        # Downsample HR lag buffer to get LR ground truth lag buffer
        # We expect downaverage_states to accept a stacked array of states and a factor.
        # If not, adapt accordingly (map over axis 0).
        # lag_hr has shape (data_lag, *unpadded_hr_shape)

        # vectorize over time axis
        lag_hr_down = downaverage_states(lag_hr, training_config.downscale_factor)  # shape (data_lag, *unpadded_lr_shape)


        # --- 4) compute gradients and update network params (if applicable) ---
        # We create a function that produces the loss given network params. In your setup the network
        # may not affect the LR sim because cnn_mhd_corrector is False; this still preserves signature.
        def loss_wrt_params(net_params):
            batch_params = params._replace(
                cnn_mhd_corrector_params=params.cnn_mhd_corrector_params._replace(
                    network_params=net_params
                )
            )
            state_lr_after, time_lr_after, lag_lr = run_for_lag_steps(
                state_lr, current_time, config_lr, batch_params, save_full_sim, helper_data_lr
            )
            return loss_function(lag_lr, lag_hr_down), (
                state_lr_after,
                time_lr_after,
            )
        # Compute gradients; filter_value_and_grad will return grads in same pytree layout as network_params
        (loss_val, (state_lr_after, time_lr_after)), grads = eqx.filter_value_and_grad(loss_wrt_params, has_aux=True)(network_params)
        updates, new_opt_state = optimizer.update(grads, opt_state_local, network_params)
        new_network_params = eqx.apply_updates(network_params, updates)

        # prepare carry for next batch:
        return (
            (
                state_hr_after,
                state_lr_after,
                time_hr_after,  # both advanced by same dt steps
                new_network_params,
                new_opt_state,
            ),
            loss_val,
        )

    final_carry, losses = jax.lax.scan(
        batch_step, batch_init_carry, jnp.arange(num_batches)
    )

    (
        final_state_hr,
        final_state_lr,
        final_time,
        final_network_params,
        final_opt_state,
    ) = final_carry

    # return losses per batch and the final objects. Choose which sim_data to return (I return HR here).
    return losses, final_network_params, final_opt_state