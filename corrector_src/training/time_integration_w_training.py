import jax
import jax.numpy as jnp
from typing import Tuple

# type checking
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from typing import Union

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

import optax
import equinox as eqx

import jax.tree_util as jtu


def time_integration_training_and_saving(
    initial_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    training_config: TrainingConfig,
    target_data: STATE_TYPE,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    train_early_steps: bool = True,
) -> Tuple[jnp.ndarray, eqx.Module, optax.OptState, SnapshotData]:
    """
    Use JAX scan to run simulation with step-by-step training using last data_lag states. Returns the full simulation as well

    Returns:
        losses: Array of losses at each step (starting from data_lag)
        final_network_params: Final trained network parameters
        full_sim_data: Full simulation data
    """

    total_steps = config.num_checkpoints  # not to confuse with num snapshots, snapshots is the ammount of states that are returned, checkpoints is the ammount of times the evolve loop is called
    data_lag = training_config.n_look_behind  # renamed from n_look_behind

    # we must pad the state with ghost cells
    if config.geometry == CARTESIAN:
        original_shape = initial_state.shape

        if config.dimensionality == 1:
            initial_state = jnp.pad(initial_state, ((0, 0), (2, 2)), mode="edge")
        elif config.dimensionality == 2:
            initial_state = jnp.pad(
                initial_state, ((0, 0), (2, 2), (2, 2)), mode="edge"
            )
        elif config.dimensionality == 3:
            initial_state = jnp.pad(
                initial_state, ((0, 0), (2, 2), (2, 2), (2, 2)), mode="edge"
            )

    def initialize_full_sim_data():
        # Initialize simulation data storage
        full_time_points = jnp.zeros(total_steps)
        full_states = jnp.zeros((total_steps, *original_shape))
        full_total_mass = jnp.zeros(total_steps)
        full_total_energy = jnp.zeros(total_steps)
        full_internal_energy = jnp.zeros(total_steps)
        full_kinetic_energy = jnp.zeros(total_steps)

        if config.self_gravity:
            full_gravitational_energy = jnp.zeros(total_steps)
        else:
            full_gravitational_energy = None

        full_sim_data = SnapshotData(
            time_points=full_time_points,
            states=full_states,
            total_mass=full_total_mass,
            total_energy=full_total_energy,
            internal_energy=full_internal_energy,
            kinetic_energy=full_kinetic_energy,
            gravitational_energy=full_gravitational_energy,
            current_checkpoint=0,
        )
        return full_sim_data

    full_sim_data = initialize_full_sim_data()
    # Initialize neural network parameters and optimizer
    neural_net_params = params.cnn_mhd_corrector_params.network_params

    # Initialize losses array (only computed after data_lag steps)
    num_loss_steps = total_steps - data_lag
    losses = jnp.zeros(num_loss_steps)
    lag_data = jnp.zeros((data_lag, *original_shape))

    def unpad_state(state):
        """Helper function to remove ghost cells."""
        if config.geometry == CARTESIAN:
            if config.dimensionality == 1:
                return jax.lax.slice_in_dim(state, 2, state.shape[1] - 2, axis=1)
            elif config.dimensionality == 2:
                unpad_state = jax.lax.slice_in_dim(state, 2, state.shape[1] - 2, axis=1)
                return jax.lax.slice_in_dim(
                    unpad_state, 2, unpad_state.shape[2] - 2, axis=2
                )
            elif config.dimensionality == 3:
                unpad_state = jax.lax.slice_in_dim(state, 2, state.shape[1] - 2, axis=1)
                unpad_state = jax.lax.slice_in_dim(
                    unpad_state, 2, unpad_state.shape[2] - 2, axis=2
                )
                return jax.lax.slice_in_dim(
                    unpad_state, 2, unpad_state.shape[3] - 2, axis=3
                )
        return state

    def update_simulation_data(time, state, sim_data, step_idx):
        """Update simulation data at given step."""
        unpadded_state = unpad_state(state)

        time_points = sim_data.time_points.at[step_idx].set(time)
        states = sim_data.states.at[step_idx].set(unpadded_state)

        total_mass = sim_data.total_mass.at[step_idx].set(
            calculate_total_mass(unpadded_state, helper_data, config)
        )
        total_energy = sim_data.total_energy.at[step_idx].set(
            calculate_total_energy(
                unpadded_state,
                helper_data,
                params.gamma,
                params.gravitational_constant,
                config,
                registered_variables,
            )
        )
        internal_energy = sim_data.internal_energy.at[step_idx].set(
            calculate_internal_energy(
                unpadded_state,
                helper_data,
                params.gamma,
                config,
                registered_variables,
            )
        )
        kinetic_energy = sim_data.kinetic_energy.at[step_idx].set(
            calculate_kinetic_energy(
                unpadded_state, helper_data, config, registered_variables
            )
        )

        if config.self_gravity:
            gravitational_energy = sim_data.gravitational_energy.at[step_idx].set(
                calculate_gravitational_energy(
                    unpadded_state,
                    helper_data,
                    params.gravitational_constant,
                    config,
                    registered_variables,
                )
            )
        else:
            gravitational_energy = sim_data.gravitational_energy

        current_checkpoint = step_idx + 1

        return sim_data._replace(
            time_points=time_points,
            states=states,
            current_checkpoint=current_checkpoint,
            total_mass=total_mass,
            total_energy=total_energy,
            internal_energy=internal_energy,
            kinetic_energy=kinetic_energy,
            gravitational_energy=gravitational_energy,
        )

    def train_evolve_step(carry, step_idx):
        (
            state,
            time,
            sim_data,
            network_params,
            lag_data,
            opt_state,
        ) = carry

        def evolve_loss_fn(network_params_loss, carry):
            def simulation_step(carry, step_idx, network_params_sim):
                """Single simulation step with optional training."""
                (
                    state,
                    current_time,
                    sim_data,
                    _,
                    lag_data,
                    *_,
                ) = carry

                # Update simulation data first
                sim_data = update_simulation_data(
                    current_time, state, sim_data, step_idx
                )
                updated_params = params._replace(
                    cnn_mhd_corrector_params=params.cnn_mhd_corrector_params._replace(
                        network_params=network_params_sim
                    )
                )

                # Compute timestep
                if not config.fixed_timestep:
                    if config.source_term_aware_timestep:
                        dt = jax.lax.stop_gradient(
                            _source_term_aware_time_step(
                                state,
                                config,
                                updated_params,
                                helper_data,
                                registered_variables,
                                current_time,
                            )
                        )
                    else:
                        dt = jax.lax.stop_gradient(
                            _cfl_time_step(
                                state,
                                config.grid_spacing,
                                params.dt_max,
                                params.gamma,
                                config,
                                registered_variables,
                                params.C_cfl,
                            )
                        )
                else:
                    dt = jnp.asarray(params.t_end / config.num_timesteps)

                state = _run_physics_modules(
                    state,
                    dt,
                    config,
                    updated_params,
                    helper_data,
                    registered_variables,
                    current_time + dt,
                )

                # update state
                state = _evolve_state(
                    state,
                    dt,
                    updated_params.gamma,
                    updated_params.gravitational_constant,
                    config,
                    helper_data,
                    registered_variables,
                )
                current_time += dt

                # updating lag data
                def update_lag_data(state, lag_data, step_idx):
                    def roll_update_lag_data(state, lag_data):
                        """roll lag data and change the last state for the new one"""
                        unpadded_state = unpad_state(state)
                        lag_data = jnp.roll(lag_data, -1, axis=0)
                        lag_data = lag_data.at[-1].set(unpadded_state)

                        return lag_data

                    def update_lag_data(state, lag_data):
                        """change the last state for the new one"""
                        unpadded_state = unpad_state(state)

                        lag_data = lag_data.at[step_idx].set(unpadded_state)

                        return lag_data

                    lag_data = jax.lax.cond(
                        step_idx >= data_lag,
                        roll_update_lag_data,
                        update_lag_data,
                        state,
                        lag_data,
                    )
                    return lag_data

                lag_data = update_lag_data(state, lag_data, step_idx)

                return (
                    state,
                    current_time,
                    sim_data,
                    network_params_sim,
                    lag_data,
                )

            state, current_time, sim_data, network_params_loss, lag_data = (
                simulation_step(carry, step_idx, network_params_loss)
            )

            # extract data_lag slices from the gt
            end_idx = step_idx + 1
            start_idx = jnp.maximum(0, end_idx - data_lag)
            actual_length = end_idx - start_idx
            jax.debug.print("step_idx {} end_idx {} actual_length {}",
                step_idx,
                end_idx,
                actual_length)
            
            ground_truth_states = jax.lax.dynamic_slice_in_dim(
                target_data, start_idx, data_lag, axis=0
            )
            """
            jax.debug.print(
                "step_idx {} end_idx {} actual_length {}",
                step_idx,
                end_idx,
                actual_length,
            )
            """
            mask = jnp.arange(data_lag) < actual_length

            def masked_loss_function(pred_states, gt_states, mask, training_config):
                expanded_mask = mask.reshape(-1, *([1] * (pred_states.ndim - 1)))

                masked_pred = pred_states * expanded_mask
                masked_gt = gt_states * expanded_mask

                loss = training_config.loss_function(
                    masked_pred, masked_gt, training_config
                )
                return loss

            loss = jax.lax.cond(
                train_early_steps & (actual_length < data_lag),
                lambda: masked_loss_function(
                    lag_data, ground_truth_states, mask, training_config
                ),
                lambda: training_config.loss_function(
                    lag_data, ground_truth_states, training_config
                ),
            )

            return loss

        loss_value, grads = eqx.filter_value_and_grad(evolve_loss_fn)(
            network_params, carry
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, network_params)
        network_params_new = eqx.apply_updates(network_params, updates)
        """
        jax.debug.print("Loss: {}", loss_value)
        grad_norm = jnp.sqrt(
            sum([jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grads)])
        )
        jax.debug.print("Grad norm: {}", grad_norm)
        """
        """
        delta = jax.tree_util.tree_map(
            lambda p_new, p_old: jnp.sum((p_new - p_old) ** 2),
            network_params_new,
            network_params,
        )
        jax.debug.print("Param change: {}", sum(jax.tree_util.tree_leaves(delta)))
        """

        return (
            state,
            time,
            sim_data,
            network_params_new,
            lag_data,
            opt_state_new,
        ), loss_value

    initial_carry = (
        initial_state,
        0.0,
        full_sim_data,
        neural_net_params,
        lag_data,
        opt_state,
    )

    (
        (
            final_state,
            final_time,
            final_sim_data,
            final_network_params,
            _,
            final_opt_state,
        ),
        losses,
    ) = jax.lax.scan(train_evolve_step, initial_carry, jnp.arange(total_steps))

    return losses, final_network_params, final_opt_state, final_sim_data


def time_integration_training(
    initial_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    training_config: TrainingConfig,
    target_data: STATE_TYPE,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    train_early_steps: bool = True,
) -> Tuple[jnp.ndarray, eqx.Module, optax.OptState]:
    """
    Use JAX scan to run simulation with step-by-step training using last data_lag states.

    Returns:
        losses: Array of losses at each step (starting from data_lag)
        final_network_params: Final trained network parameters
        full_sim_data: Full simulation data
    """

    total_steps = config.num_checkpoints  # not to confuse with num snapshots, snapshots is the ammount of states that are returned, checkpoints is the ammount of times the evolve loop is called
    data_lag = training_config.n_look_behind  # renamed from n_look_behind

    # we must pad the state with ghost cells
    if config.geometry == CARTESIAN:
        original_shape = initial_state.shape

        if config.dimensionality == 1:
            initial_state = jnp.pad(initial_state, ((0, 0), (2, 2)), mode="edge")
        elif config.dimensionality == 2:
            initial_state = jnp.pad(
                initial_state, ((0, 0), (2, 2), (2, 2)), mode="edge"
            )
        elif config.dimensionality == 3:
            initial_state = jnp.pad(
                initial_state, ((0, 0), (2, 2), (2, 2), (2, 2)), mode="edge"
            )

    # Initialize neural network parameters and optimizer
    neural_net_params = params.cnn_mhd_corrector_params.network_params

    # Initialize losses array (only computed after data_lag steps)
    num_loss_steps = total_steps - data_lag
    losses = jnp.zeros(num_loss_steps)
    lag_data = jnp.zeros((data_lag, *original_shape))

    def unpad_state(state):
        """Helper function to remove ghost cells."""
        if config.geometry == CARTESIAN:
            if config.dimensionality == 1:
                return jax.lax.slice_in_dim(state, 2, state.shape[1] - 2, axis=1)
            elif config.dimensionality == 2:
                unpad_state = jax.lax.slice_in_dim(state, 2, state.shape[1] - 2, axis=1)
                return jax.lax.slice_in_dim(
                    unpad_state, 2, unpad_state.shape[2] - 2, axis=2
                )
            elif config.dimensionality == 3:
                unpad_state = jax.lax.slice_in_dim(state, 2, state.shape[1] - 2, axis=1)
                unpad_state = jax.lax.slice_in_dim(
                    unpad_state, 2, unpad_state.shape[2] - 2, axis=2
                )
                return jax.lax.slice_in_dim(
                    unpad_state, 2, unpad_state.shape[3] - 2, axis=3
                )
        return state

    def train_evolve_step(carry, step_idx):
        (
            state,
            time,
            network_params,
            lag_data,
            opt_state,
        ) = carry

        def evolve_loss_fn(network_params_loss, carry):
            def simulation_step(carry, step_idx, network_params_sim):
                """Single simulation step with optional training."""
                (
                    state,
                    current_time,
                    _,
                    lag_data,
                    *_,
                ) = carry

                # Update simulation data first
                updated_params = params._replace(
                    cnn_mhd_corrector_params=params.cnn_mhd_corrector_params._replace(
                        network_params=network_params_sim
                    )
                )

                # Compute timestep
                if not config.fixed_timestep:
                    if config.source_term_aware_timestep:
                        dt = jax.lax.stop_gradient(
                            _source_term_aware_time_step(
                                state,
                                config,
                                updated_params,
                                helper_data,
                                registered_variables,
                                current_time,
                            )
                        )
                    else:
                        dt = jax.lax.stop_gradient(
                            _cfl_time_step(
                                state,
                                config.grid_spacing,
                                params.dt_max,
                                params.gamma,
                                config,
                                registered_variables,
                                params.C_cfl,
                            )
                        )
                else:
                    dt = jnp.asarray(params.t_end / config.num_timesteps)

                state = _run_physics_modules(
                    state,
                    dt,
                    config,
                    updated_params,
                    helper_data,
                    registered_variables,
                    current_time + dt,
                )

                # update state
                state = _evolve_state(
                    state,
                    dt,
                    updated_params.gamma,
                    updated_params.gravitational_constant,
                    config,
                    helper_data,
                    registered_variables,
                )
                current_time += dt

                # updating lag data
                def update_lag_data(state, lag_data, step_idx):
                    def roll_update_lag_data(state, lag_data):
                        """roll lag data and change the last state for the new one"""
                        unpadded_state = unpad_state(state)
                        lag_data = jnp.roll(lag_data, -1, axis=0)
                        lag_data = lag_data.at[-1].set(unpadded_state)

                        return lag_data

                    def update_lag_data(state, lag_data):
                        """change the last state for the new one"""
                        unpadded_state = unpad_state(state)

                        lag_data = lag_data.at[step_idx].set(unpadded_state)

                        return lag_data

                    lag_data = jax.lax.cond(
                        step_idx >= data_lag,
                        roll_update_lag_data,
                        update_lag_data,
                        state,
                        lag_data,
                    )
                    return lag_data

                lag_data = update_lag_data(state, lag_data, step_idx)

                return (
                    state,
                    current_time,
                    network_params_sim,
                    lag_data,
                )

            state, current_time, network_params_loss, lag_data = simulation_step(
                carry, step_idx, network_params_loss
            )

            # extract data_lag slices from the gt
            end_idx = step_idx + 1
            start_idx = jnp.maximum(0, end_idx - data_lag)
            actual_length = end_idx - start_idx

            ground_truth_states = jax.lax.dynamic_slice_in_dim(
                target_data, start_idx, data_lag, axis=0
            )
            """
            jax.debug.print(
                "step_idx {} end_idx {} actual_length {}",
                step_idx,
                end_idx,
                actual_length,
            )
            """
            mask = jnp.arange(data_lag) < actual_length

            def masked_loss_function(pred_states, gt_states, mask, training_config):
                expanded_mask = mask.reshape(-1, *([1] * (pred_states.ndim - 1)))

                masked_pred = pred_states * expanded_mask
                masked_gt = gt_states * expanded_mask

                loss = training_config.loss_function(
                    masked_pred, masked_gt, training_config
                )
                return loss

            loss = jax.lax.cond(
                train_early_steps & (actual_length < data_lag),
                lambda: masked_loss_function(
                    lag_data, ground_truth_states, mask, training_config
                ),
                lambda: training_config.loss_function(
                    lag_data, ground_truth_states, training_config
                ),
            )

            return loss

        loss_value, grads = eqx.filter_value_and_grad(evolve_loss_fn)(
            network_params, carry
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, network_params)
        network_params_new = eqx.apply_updates(network_params, updates)
        """
        jax.debug.print("Loss: {}", loss_value)
        grad_norm = jnp.sqrt(
            sum([jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grads)])
        )
        jax.debug.print("Grad norm: {}", grad_norm)
        """
        """
        delta = jax.tree_util.tree_map(
            lambda p_new, p_old: jnp.sum((p_new - p_old) ** 2),
            network_params_new,
            network_params,
        )
        jax.debug.print("Param change: {}", sum(jax.tree_util.tree_leaves(delta)))
        """

        return (
            state,
            time,
            network_params_new,
            lag_data,
            opt_state_new,
        ), loss_value

    initial_carry = (
        initial_state,
        0.0,
        neural_net_params,
        lag_data,
        opt_state,
    )

    (
        (
            final_state,
            final_time,
            final_network_params,
            _,
            final_opt_state,
        ),
        losses,
    ) = jax.lax.scan(train_evolve_step, initial_carry, jnp.arange(total_steps))

    return losses, final_network_params, final_opt_state