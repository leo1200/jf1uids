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

# timing
from timeit import default_timer as timer

from corrector_src.training.training_config import TrainingConfig
from corrector_src.model._cnn_mhd_corrector_options import (
    CNNMHDParams,
    CNNMHDconfig,
)

import optax
import equinox as eqx


def step_based_training_with_losses(
    initial_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    training_config: TrainingConfig,
    target_data: STATE_TYPE,
    train_early_steps: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, SnapshotData]:
    """
    Use JAX scan to run simulation with step-by-step training using last n_look_ahead states.

    Returns:
        losses: Array of losses at each step (starting from n_look_ahead)
        final_network_params: Final trained network parameters
        full_sim_data: Full simulation data
    """

    total_steps = config.num_checkpoints
    n_look_ahead = training_config.n_look_behind  # renamed from n_look_behind

    # print(f"Total steps: {total_steps}, n_look_ahead: {n_look_ahead}")

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

    # Initialize neural network parameters and optimizer
    neural_net_params = params.cnn_mhd_corrector_params.network_params
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(neural_net_params)

    # Initialize losses array (only computed after n_look_ahead steps)
    num_loss_steps = total_steps - n_look_ahead
    losses = jnp.zeros(num_loss_steps)

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

    def simulation_step(carry, step_idx):
        """Single simulation step with optional training."""
        (
            state,
            current_time,
            sim_data,
            network_params_current,
            opt_state_current,
            losses_array,
        ) = carry

        # Update simulation data first
        sim_data = update_simulation_data(current_time, state, sim_data, step_idx)

        # Compute timestep
        if not config.fixed_timestep:
            if config.source_term_aware_timestep:
                dt = jax.lax.stop_gradient(
                    _source_term_aware_time_step(
                        state,
                        config,
                        params._replace(
                            cnn_mhd_corrector_params=params.cnn_mhd_corrector_params._replace(
                                network_params=network_params_current
                            )
                        ),
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

        # Physics and evolution step
        updated_params = params._replace(
            cnn_mhd_corrector_params=params.cnn_mhd_corrector_params._replace(
                network_params=network_params_current
            )
        )

        state = _run_physics_modules(
            state,
            dt,
            config,
            updated_params,
            helper_data,
            registered_variables,
            current_time + dt,
        )
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

        # Training step
        def do_training():
            """Perform training using available states."""

            def loss_fn(network_params_for_loss):
                """Compute loss using available simulated states."""
                # Always slice the maximum possible window (n_look_ahead)
                # For early steps, we'll mask out the unused portions

                end_idx = step_idx + 1  # Current position after updating sim_data
                start_idx = jnp.maximum(0, end_idx - n_look_ahead)
                actual_length = end_idx - start_idx

                # Always extract n_look_ahead states, padding with zeros if necessary
                predicted_states = jax.lax.dynamic_slice_in_dim(
                    sim_data.states, start_idx, n_look_ahead, axis=0
                )
                ground_truth_states = jax.lax.dynamic_slice_in_dim(
                    target_data, start_idx, n_look_ahead, axis=0
                )

                # Create mask for valid states
                mask = jnp.arange(n_look_ahead) < actual_length

                # Apply mask to loss computation - only compute loss on valid states
                def masked_loss_function(pred_states, gt_states, mask, training_config):
                    # Expand mask to match state dimensions
                    expanded_mask = mask.reshape(-1, *([1] * (pred_states.ndim - 1)))

                    # Zero out invalid states
                    masked_pred = pred_states * expanded_mask
                    masked_gt = gt_states * expanded_mask

                    # Compute loss only on valid timesteps
                    loss = training_config.loss_function(
                        masked_pred, masked_gt, training_config
                    )

                    return loss

                # Only apply masking for early training steps
                loss = jax.lax.cond(
                    train_early_steps & (actual_length < n_look_ahead),
                    lambda: masked_loss_function(
                        predicted_states, ground_truth_states, mask, training_config
                    ),
                    lambda: training_config.loss_function(
                        predicted_states, ground_truth_states, training_config
                    ),
                )

                return loss

            def train_step(network_params_arrays, opt_state_local):
                """Performs one step of gradient descent."""
                loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
                    network_params_arrays
                )
                updates, opt_state_new = optimizer.update(
                    grads, opt_state_local, network_params_arrays
                )
                network_params_new = eqx.apply_updates(network_params_arrays, updates)
                return network_params_new, opt_state_new, loss_value

            return train_step(network_params_current, opt_state_current)

        def no_training():
            """Skip training step."""
            return network_params_current, opt_state_current, 0.0

        # Determine whether to train at this step
        should_train = jax.lax.cond(
            train_early_steps,
            lambda: True,  # Always train if train_early_steps is True
            lambda: step_idx >= n_look_ahead,  # Only train after n_look_ahead if False
        )

        network_params_new, opt_state_new, loss_value = jax.lax.cond(
            should_train, do_training, no_training
        )

        # Update losses array
        def update_losses():
            if train_early_steps:
                loss_idx = step_idx
            else:
                loss_idx = step_idx - n_look_ahead
            return losses_array.at[loss_idx].set(loss_value)

        def keep_losses():
            return losses_array

        losses_array = jax.lax.cond(should_train, update_losses, keep_losses)

        return (
            state,
            current_time,
            sim_data,
            network_params_new,
            opt_state_new,
            losses_array,
        ), None

    # Run simulation with training
    initial_carry = (
        initial_state,
        0.0,
        full_sim_data,
        neural_net_params,
        opt_state,
        losses,
    )

    (
        (
            final_state,
            final_time,
            final_sim_data,
            final_network_params,
            final_opt_state,
            final_losses,
        ),
        _,
    ) = jax.lax.scan(simulation_step, initial_carry, jnp.arange(total_steps))

    return final_losses, final_network_params, final_sim_data
