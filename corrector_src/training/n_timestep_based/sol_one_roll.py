"""time integration with nn training
each new snapshot is saved onto a lag_data which takes that snapshot and the last data_lag snapshots
with those and the reference ground truth the loss is computed and the optimization is called
in practice doesnt work due to several reasons:
    to get the same states to the ground truth the fixed timestep is used
    as we are comparing snapshot to snapshot is very memory requiring
after some proper anlyisis this approach is basically the same as the n_timetstep_based with the difference of rolling the snapshot data
which makes it even more computationaly costly and expensive"""

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

import optax
import equinox as eqx

from jax.experimental import checkify

from functools import partial


def time_integration_training(
    initial_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    training_config: TrainingConfig,
    target_data: STATE_TYPE,
    optimizer: optax.GradientTransformation,
    loss_function: Callable[[jnp.ndarray, jnp.ndarray, TrainingConfig], jnp.ndarray],
    opt_state: optax.OptState,
    save_full_sim: bool = False,
) -> Tuple[jnp.ndarray, eqx.Module, optax.OptState, SnapshotData]:
    if config.runtime_debugging:
        checked_integration = checkify.checkify(_time_integration_training)

        err, final_state = checked_integration(
            initial_state,
            config,
            params,
            helper_data,
            registered_variables,
            training_config,
            target_data,
            optimizer,
            loss_function,
            opt_state,
            save_full_sim,
        )

        return err, final_state

    else:
        return _time_integration_training(
            initial_state,
            config,
            params,
            helper_data,
            registered_variables,
            training_config,
            target_data,
            optimizer,
            loss_function,
            opt_state,
            save_full_sim,
        )


@partial(
    jax.jit,
    static_argnames=[
        "config",
        "registered_variables",
        "training_config",
        "optimizer",
        "save_full_sim",
        "loss_function",
    ],
)
def _time_integration_training(
    initial_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    training_config: TrainingConfig,
    target_data: STATE_TYPE,
    optimizer: optax.GradientTransformation,
    loss_function: Callable[[jnp.ndarray, jnp.ndarray, TrainingConfig], jnp.ndarray],
    opt_state: optax.OptState,
    save_full_sim: bool = False,
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
    num_batches = total_steps // data_lag
    losses = jnp.zeros(num_batches)

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

    def update_simulation_data(time, state, sim_data):
        """Update simulation data at given step."""
        unpadded_state = unpad_state(state)
        current_checkpoint = sim_data.current_checkpoint

        time_points = sim_data.time_points.at[current_checkpoint].set(time)
        states = sim_data.states.at[current_checkpoint].set(unpadded_state)

        total_mass = sim_data.total_mass.at[current_checkpoint].set(
            calculate_total_mass(unpadded_state, helper_data, config)
        )
        total_energy = sim_data.total_energy.at[current_checkpoint].set(
            calculate_total_energy(
                unpadded_state,
                helper_data,
                params.gamma,
                params.gravitational_constant,
                config,
                registered_variables,
            )
        )
        internal_energy = sim_data.internal_energy.at[current_checkpoint].set(
            calculate_internal_energy(
                unpadded_state,
                helper_data,
                params.gamma,
                config,
                registered_variables,
            )
        )
        kinetic_energy = sim_data.kinetic_energy.at[current_checkpoint].set(
            calculate_kinetic_energy(
                unpadded_state, helper_data, config, registered_variables
            )
        )

        if config.self_gravity:
            gravitational_energy = sim_data.gravitational_energy.at[
                current_checkpoint
            ].set(
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

        current_checkpoint = current_checkpoint + 1

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

    def not_update_simulation_data(time, state, sim_data):
        return sim_data

    def batch_step(batch_carry, batch_i):
        (
            state,
            time,
            sim_data,
            network_params,
            opt_state,
        ) = batch_carry

        def evolve_loss_fn(batch_network_params, loss_carry):
            # unsure if it should be in the carry
            (
                batch_initial_state,
                batch_initial_time,
                sim_data,
            ) = loss_carry

            batch_params = params._replace(
                cnn_mhd_corrector_params=params.cnn_mhd_corrector_params._replace(
                    network_params=batch_network_params
                )
            )

            def simulation_step(sim_carry, step_idx):
                """Single simulation step called data_lag times
                step_idx [0, data_lag]"""
                (
                    state,
                    current_time,
                    sim_data,
                    lag_data,
                ) = sim_carry

                # Update simulation data first
                sim_data = jax.lax.cond(
                    save_full_sim,
                    update_simulation_data,
                    not_update_simulation_data,
                    current_time,
                    state,
                    sim_data,
                )

                # Compute timestep
                if not config.fixed_timestep:
                    if config.source_term_aware_timestep:
                        dt = jax.lax.stop_gradient(
                            _source_term_aware_time_step(
                                state,
                                config,
                                batch_params,
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
                    batch_params,
                    helper_data,
                    registered_variables,
                    current_time + dt,
                )
                """"
                if config.runtime_debugging:
                    jax.debug.callback(
                        lambda s, idx: print(f"Step {idx} after physics: has NaN = {jnp.isnan(s).any()}"),
                        state, step_idx
                    )                # update state
                """
                state = _evolve_state(
                    state,
                    dt,
                    batch_params.gamma,
                    batch_params.gravitational_constant,
                    config,
                    helper_data,
                    registered_variables,
                )
                current_time += dt
                unpadded_state = unpad_state(state)
                lag_data = lag_data.at[step_idx].set(unpadded_state)
                """
                if config.runtime_debugging:
                    jax.debug.callback(
                        lambda s, idx: print(f"Step {idx} after physics: has NaN = {jnp.isnan(s).any()}"),
                        state, step_idx
                    )                # update state
                checkify.check(
                    jnp.isfinite(state).all(),
                    "Non-finite value in state after _evolve_state",
                )
                """
                return (
                    state,
                    current_time,
                    sim_data,
                    lag_data,
                ), None

            lag_data = jnp.zeros((data_lag, *original_shape))
            inital_batch_carry = (
                batch_initial_state,
                batch_initial_time,
                sim_data,
                lag_data,
            )
            final_batch_carry, _ = jax.lax.scan(
                simulation_step,
                inital_batch_carry,
                jnp.arange(data_lag),
            )
            (
                state,
                time,
                sim_data,
                lag_data,
            ) = final_batch_carry

            # extract data_lag slices from the gt
            start_idx = batch_i * data_lag
            # end_idx = (batch_i + 1) * data_lag

            # jax.debug.print(
            #    "step_idx {} end_idx {} batch_i {}", start_idx, end_idx, batch_i
            # )

            ground_truth_states = jax.lax.dynamic_slice_in_dim(
                target_data, start_idx, data_lag, axis=0
            )

            loss = loss_function(lag_data, ground_truth_states)
            return loss

        loss_carry = (
            state,
            time,
            sim_data,
        )

        loss_value, grads = eqx.filter_value_and_grad(evolve_loss_fn)(
            network_params, loss_carry
        )
        """
        if config.runtime_debugging:

            def grad_norms(grads):
                Compute per-leaf and total gradient L2 norms
                leaves, _ = jax.tree.flatten(grads)
                norms = [jnp.linalg.norm(g) for g in leaves if g is not None]
                total_norm = jnp.sqrt(sum(n**2 for n in norms))
                return norms, total_norm

            norms, total_norm = grad_norms(grads)
            jax.debug.print(
                "Grad norms: {} ... total: {}", norms[:5], total_norm
            )  # show first few
        """
        updates, opt_state_new = optimizer.update(grads, opt_state, network_params)
        network_params_new = eqx.apply_updates(network_params, updates)
        return (
            state,
            time,
            sim_data,
            network_params_new,
            opt_state_new,
        ), loss_value

    initial_carry = (
        initial_state,
        0.0,
        full_sim_data,
        neural_net_params,
        opt_state,
    )

    (
        (
            final_state,
            final_time,
            final_sim_data,
            final_network_params,
            final_opt_state,
        ),
        losses,
    ) = jax.lax.scan(batch_step, initial_carry, jnp.arange(num_batches))

    return losses, final_network_params, final_opt_state, final_sim_data
