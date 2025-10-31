# general
from types import NoneType
import jax
import jax.numpy as jnp
from functools import partial

from equinox.internal._loop.checkpointed import checkpointed_while_loop

# type checking
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from typing import Union, Optional, Callable, Tuple

# runtime debugging
from jax.experimental import checkify

# jf1uids constants
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    CARTESIAN,
    CYLINDRICAL,
    FORWARDS,
    SPHERICAL,
    STATE_TYPE,
)

# jf1uids containers
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.data_classes.simulation_helper_data import HelperData, get_helper_data
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
    calculate_radial_momentum,
    calculate_total_mass,
)
from jf1uids.fluid_equations.total_quantities import (
    calculate_total_energy,
    calculate_kinetic_energy,
    calculate_gravitational_energy,
)

# progress bar
from jf1uids.time_stepping._progress_bar import _show_progress

# timing
from timeit import default_timer as timer

from jf1uids.time_stepping._utils import _pad, _unpad

from corrector_src.training.training_config import TrainingConfig, TrainingParams
import optax
import equinox as eqx


def time_integration(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    optimizer: optax.GradientTransformation,
    loss_function: Callable[
        [
            jnp.ndarray,
            jnp.ndarray,
            SimulationConfig,
            RegisteredVariables,
            SimulationParams,
        ],
        jnp.ndarray,
    ],
    opt_state: optax.OptState,
    target_data: jnp.ndarray,
    snapshot_callable=None,
    training_config: Optional[TrainingConfig] = None,
    training_params: Optional[TrainingParams] = None,
    sharding: Optional[jax.NamedSharding] = None,
) -> Union[STATE_TYPE, SnapshotData]:
    """
    Integrate the fluid equations in time. For the options of
    the time integration see the simulation configuration and
    the simulation parameters.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots)
        either the final state of the fluid after the time
        integration of snapshots of the time evolution.

    """

    helper_data_pad = get_helper_data(config, sharding, padded=True)

    if config.runtime_debugging:
        errors = (
            checkify.user_checks
            | checkify.index_checks
            | checkify.float_checks
            | checkify.nan_checks
            | checkify.div_checks
        )
        checked_integration = checkify.checkify(_time_integration, errors)

        err, (losses, network_params, opt_state, state) = checked_integration(
            primitive_state,
            config,
            params,
            helper_data,
            helper_data_pad,
            registered_variables,
            optimizer,
            loss_function,
            opt_state,
            target_data,
            snapshot_callable,
            training_config,
            training_params,
        )
        err.throw()

    else:
        if config.memory_analysis:
            compiled_step = _time_integration.lower(
                primitive_state,
                config,
                params,
                helper_data,
                helper_data_pad,
                registered_variables,
                optimizer,
                loss_function,
                opt_state,
                target_data,
                snapshot_callable,
                training_config,
                training_params,
            ).compile()
            compiled_stats = compiled_step.memory_analysis()
            if compiled_stats is not None:
                # Calculate total memory usage including temporary storage, arguments, and outputs
                # Subtract alias size to avoid double-counting memory shared between different components
                total = (
                    compiled_stats.temp_size_in_bytes
                    + compiled_stats.argument_size_in_bytes
                    + compiled_stats.output_size_in_bytes
                    - compiled_stats.alias_size_in_bytes
                )
                print("=== Compiled memory usage PER DEVICE ===")
                print(
                    f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**2):.2f} MB"
                )
                print(
                    f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**2):.2f} MB"
                )
                print(f"Total size: {total / (1024**2):.2f} MB")
                print("========================================")

        losses, network_params, opt_state, state = _time_integration(
            primitive_state,
            config,
            params,
            helper_data,
            helper_data_pad,
            registered_variables,
            optimizer,
            loss_function,
            opt_state,
            target_data,
            snapshot_callable,
            training_config,
            training_params,
        )

    return losses, network_params, opt_state, state


@partial(
    jax.jit,
    static_argnames=[
        "config",
        "registered_variables",
        "snapshot_callable",
        "training_config",
        "loss_function",
        "optimizer",
    ],
)
def _time_integration(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    helper_data_pad: HelperData,
    registered_variables: RegisteredVariables,
    optimizer: optax.GradientTransformation,
    loss_function: Callable[
        [
            jnp.ndarray,
            jnp.ndarray,
            SimulationConfig,
            RegisteredVariables,
            SimulationParams,
        ],
        jnp.ndarray,
    ],
    opt_state: optax.OptState,
    target_data: jnp.ndarray,
    snapshot_callable=None,
    training_config: Optional[TrainingConfig] = None,
    training_params: Optional[TrainingParams] = None,
) -> Tuple[jnp.ndarray, eqx.Module, optax.OptState, SnapshotData]:
    """
    Time integration.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either the final state of the fluid
        after the time integration of snapshots of the time evolution.
    """

    # we must pad the state with ghost cells
    # pad the primitive state with two ghost cells on each side
    # to account for the periodic boundary conditions
    original_shape = primitive_state.shape

    primitive_state = _pad(primitive_state, config)

    # important for active boundaries influencing the time step criterion
    # for now only gas state
    if config.mhd:
        primitive_state = primitive_state.at[:-3, ...].set(
            _boundary_handler(primitive_state[:-3, ...], config)
        )
    else:
        primitive_state = _boundary_handler(primitive_state, config)

    if config.return_snapshots:
        time_points = jnp.zeros(config.num_snapshots)

        states = (
            jnp.zeros((config.num_snapshots, *original_shape))
            if config.snapshot_settings.return_states
            else None
        )
        total_mass = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_total_mass
            else None
        )
        total_energy = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_total_energy
            else None
        )
        internal_energy = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_internal_energy
            else None
        )
        kinetic_energy = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_kinetic_energy
            else None
        )
        radial_momentum = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_radial_momentum
            else None
        )

        gravitational_energy = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_gravitational_energy
            and config.self_gravity
            else None
        )

        current_checkpoint = 0

        snapshot_data = SnapshotData(
            time_points=time_points,
            states=states,
            total_mass=total_mass,
            total_energy=total_energy,
            internal_energy=internal_energy,
            kinetic_energy=kinetic_energy,
            gravitational_energy=gravitational_energy,
            current_checkpoint=current_checkpoint,
            radial_momentum=radial_momentum,
            final_state=None,
        )

    elif config.activate_snapshot_callback:
        current_checkpoint = 0
        snapshot_data = SnapshotData(
            time_points=None,
            states=None,
            total_mass=None,
            total_energy=None,
            current_checkpoint=current_checkpoint,
        )

    # def condition(carry):
    #     if config.return_snapshots or config.activate_snapshot_callback:
    #         t, _, _ = carry
    #     else:
    #         t, _ = carry
    #     return t < params.t_end

    if training_config is None:
        raise ValueError("no training config was given!!")
    else:
        tsteps_lag = config.num_timesteps / len(training_params.loss_calculation_times)

        def train_step(carry, loss_index):
            """Performs one step of gradient descent."""

            if config.return_snapshots or config.activate_snapshot_callback:
                if training_config.accumulate_grads:
                    (
                        time,
                        primitive_state,
                        snapshot_data,
                        network_params,
                        opt_state,
                        accum_grads,
                    ) = carry
                    carry_loss = (time, primitive_state, snapshot_data)
                else:
                    time, primitive_state, snapshot_data, network_params, opt_state = (
                        carry
                    )
                    carry_loss = (time, primitive_state, snapshot_data)
            else:
                if training_config.accumulate_grads:
                    time, primitive_state, network_params, opt_state, accum_grads = (
                        carry
                    )
                    carry_loss = (time, primitive_state)
                else:
                    time, primitive_state, network_params, opt_state = carry
                    carry_loss = (time, primitive_state)

            def condition_loss(carry):
                if config.return_snapshots or config.activate_snapshot_callback:
                    t, _, _ = carry
                else:
                    t, _ = carry
                return t < training_params.loss_calculation_times[loss_index]

            def loss_fn(network_parameters, carry):
                updated_params = params._replace(
                    corrector_params=params.corrector_params._replace(
                        network_params=network_parameters
                    )
                )

                def update_step(carry):
                    if config.return_snapshots:
                        time, state, snapshot_data = carry

                        def update_snapshot_data(time, state, snapshot_data):
                            time_points = snapshot_data.time_points.at[
                                snapshot_data.current_checkpoint
                            ].set(time)

                            unpad_state = _unpad(state, config)

                            if config.snapshot_settings.return_states:
                                states = snapshot_data.states.at[
                                    snapshot_data.current_checkpoint
                                ].set(unpad_state)
                            else:
                                states = None

                            if config.snapshot_settings.return_total_mass:
                                total_mass = snapshot_data.total_mass.at[
                                    snapshot_data.current_checkpoint
                                ].set(
                                    calculate_total_mass(
                                        unpad_state, helper_data, config
                                    )
                                )
                            else:
                                total_mass = None

                            if config.snapshot_settings.return_total_energy:
                                total_energy = snapshot_data.total_energy.at[
                                    snapshot_data.current_checkpoint
                                ].set(
                                    calculate_total_energy(
                                        unpad_state,
                                        helper_data,
                                        updated_params.gamma,
                                        updated_params.gravitational_constant,
                                        config,
                                        registered_variables,
                                    )
                                )
                            else:
                                total_energy = None

                            if config.snapshot_settings.return_internal_energy:
                                internal_energy = snapshot_data.internal_energy.at[
                                    snapshot_data.current_checkpoint
                                ].set(
                                    calculate_internal_energy(
                                        unpad_state,
                                        helper_data,
                                        updated_params.gamma,
                                        config,
                                        registered_variables,
                                    )
                                )
                            else:
                                internal_energy = None

                            if config.snapshot_settings.return_kinetic_energy:
                                kinetic_energy = snapshot_data.kinetic_energy.at[
                                    snapshot_data.current_checkpoint
                                ].set(
                                    calculate_kinetic_energy(
                                        unpad_state,
                                        helper_data,
                                        config,
                                        registered_variables,
                                    )
                                )
                            else:
                                kinetic_energy = None

                            if config.snapshot_settings.return_radial_momentum:
                                radial_momentum = snapshot_data.radial_momentum.at[
                                    snapshot_data.current_checkpoint
                                ].set(
                                    calculate_radial_momentum(
                                        unpad_state,
                                        helper_data,
                                        config,
                                        registered_variables,
                                    )
                                )
                            else:
                                radial_momentum = None

                            if (
                                config.self_gravity
                                and config.snapshot_settings.return_gravitational_energy
                            ):
                                gravitational_energy = (
                                    snapshot_data.gravitational_energy.at[
                                        snapshot_data.current_checkpoint
                                    ].set(
                                        calculate_gravitational_energy(
                                            unpad_state,
                                            helper_data,
                                            updated_params.gravitational_constant,
                                            config,
                                            registered_variables,
                                        )
                                    )
                                )
                            else:
                                gravitational_energy = None

                            current_checkpoint = snapshot_data.current_checkpoint + 1
                            snapshot_data = snapshot_data._replace(
                                time_points=time_points,
                                states=states,
                                current_checkpoint=current_checkpoint,
                                total_mass=total_mass,
                                total_energy=total_energy,
                                internal_energy=internal_energy,
                                kinetic_energy=kinetic_energy,
                                gravitational_energy=gravitational_energy,
                                radial_momentum=radial_momentum,
                            )
                            return snapshot_data

                        def dont_update_snapshot_data(time, state, snapshot_data):
                            return snapshot_data

                        if config.use_specific_snapshot_timepoints:
                            snapshot_data = jax.lax.cond(
                                jnp.abs(
                                    time
                                    - updated_params.snapshot_timepoints[
                                        snapshot_data.current_checkpoint
                                    ]
                                )
                                < 1e-12,
                                update_snapshot_data,
                                dont_update_snapshot_data,
                                time,
                                state,
                                snapshot_data,
                            )
                        else:
                            snapshot_data = jax.lax.cond(
                                time
                                >= snapshot_data.current_checkpoint
                                * updated_params.t_end
                                / config.num_snapshots,
                                update_snapshot_data,
                                dont_update_snapshot_data,
                                time,
                                state,
                                snapshot_data,
                            )

                        num_iterations = snapshot_data.num_iterations + 1
                        snapshot_data = snapshot_data._replace(
                            num_iterations=num_iterations
                        )

                    elif (
                        config.activate_snapshot_callback
                    ):  # for active snapshot_callback!!
                        time, state, snapshot_data = carry

                        def update_snapshot_data(snapshot_data):
                            current_checkpoint = snapshot_data.current_checkpoint + 1
                            snapshot_data = snapshot_data._replace(
                                current_checkpoint=current_checkpoint
                            )

                            jax.debug.callback(
                                snapshot_callable, time, state, registered_variables
                            )

                            return snapshot_data

                        def dont_update_snapshot_data(snapshot_data):
                            return snapshot_data

                        snapshot_data = jax.lax.cond(
                            time
                            >= snapshot_data.current_checkpoint
                            * updated_params.t_end
                            / config.num_snapshots,
                            update_snapshot_data,
                            dont_update_snapshot_data,
                            snapshot_data,
                        )

                        num_iterations = snapshot_data.num_iterations + 1
                        snapshot_data = snapshot_data._replace(
                            num_iterations=num_iterations
                        )
                    else:
                        time, state = carry

                    # do not differentiate through the choice of the time step
                    if not config.fixed_timestep:
                        if config.source_term_aware_timestep:
                            dt = jax.lax.stop_gradient(
                                _source_term_aware_time_step(
                                    state,
                                    config,
                                    updated_params,
                                    helper_data_pad,
                                    registered_variables,
                                    time,
                                )
                            )
                        else:
                            dt = jax.lax.stop_gradient(
                                _cfl_time_step(
                                    state,
                                    config.grid_spacing,
                                    updated_params.dt_max,
                                    updated_params.gamma,
                                    config,
                                    registered_variables,
                                    params.C_cfl,
                                )
                            )
                    else:
                        dt = updated_params.t_end / config.num_timesteps

                    if (
                        config.use_specific_snapshot_timepoints
                        and config.return_snapshots
                    ):
                        dt = jnp.minimum(
                            dt,
                            updated_params.snapshot_timepoints[
                                snapshot_data.current_checkpoint
                            ]
                            - time,
                        )

                    if (
                        config.exact_end_time
                        and not config.use_specific_snapshot_timepoints
                    ):
                        dt = jnp.minimum(dt, updated_params.t_end - time)

                    if training_config.exact_end_time:
                        dt = jnp.minimum(
                            dt,
                            training_params.loss_calculation_times[loss_index] - time,
                        )
                    # for now we mainly consider the stellar wind, a constant source term term,
                    # so the source is handled via a simple Euler step but generally
                    # a higher order method (in a split fashion) may be used

                    # state = _run_physics_modules(state, dt / 2, config, params, helper_data, registered_variables, time)
                    state = _run_physics_modules(
                        state,
                        dt,
                        config,
                        updated_params,
                        helper_data_pad,
                        registered_variables,
                        time + dt,
                    )
                    # jax.debug.print(
                    #     "nans in state after run_physics {nans}",
                    #     nans=jnp.any(jnp.isnan(state)),
                    # )

                    state = _evolve_state(
                        state,
                        dt,
                        updated_params.gamma,
                        updated_params.gravitational_constant,
                        config,
                        updated_params,
                        helper_data_pad,
                        registered_variables,
                    )
                    # jax.debug.print(
                    #     "nans in state after evolve_state {nans}",
                    #     nans=jnp.any(jnp.isnan(state)),
                    # )

                    time += dt

                    if (
                        config.use_specific_snapshot_timepoints
                        and config.return_snapshots
                    ):
                        snapshot_data = jax.lax.cond(
                            jnp.abs(time - updated_params.t_end) < 1e-12,
                            update_snapshot_data,
                            dont_update_snapshot_data,
                            time,
                            state,
                            snapshot_data,
                        )

                    if config.progress_bar:
                        jax.debug.callback(_show_progress, time, updated_params.t_end)

                    if config.return_snapshots or config.activate_snapshot_callback:
                        carry = (time, state, snapshot_data)
                    else:
                        carry = (time, state)

                    return carry

                def update_step_for(_, carry):
                    return update_step(carry)

                if not config.fixed_timestep:
                    if config.differentiation_mode == BACKWARDS:
                        carry = checkpointed_while_loop(
                            condition_loss,
                            update_step,
                            carry,
                            checkpoints=config.num_checkpoints,
                        )
                    elif config.differentiation_mode == FORWARDS:
                        carry = jax.lax.while_loop(condition_loss, update_step, carry)
                    else:
                        raise ValueError("Unknown differentiation mode.")
                else:
                    carry = jax.lax.fori_loop(0, tsteps_lag, update_step_for, carry)

                if config.return_snapshots or config.activate_snapshot_callback:
                    time, final_state, snapshot_data = carry
                else:
                    time, final_state = carry

                final_state = _unpad(final_state, config)
                # jax.debug.print(
                #     "nans in final state {nans}, nans in target {nans_target}, target_data_shape {shape}, loss_index {index}",
                #     nans=jnp.any(jnp.isnan(final_state)),
                #     nans_target=jnp.any(jnp.isnan(target_data[loss_index])),
                #     shape=target_data.shape,
                #     index=loss_index,
                # )
                total_loss, loss_components = loss_function(
                    target_data[loss_index],
                    final_state,
                    config,
                    registered_variables,
                    params,
                )
                return total_loss, loss_components, carry

            (total_loss, loss_components, carry_loss), grads = (
                eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                    network_params, carry_loss
                )
            )
            if training_config.accumulate_grads:
                scale = 1.0  # / len(training_params.loss_calculation_times)
                accum_grads = jax.tree_util.tree_map(
                    lambda a, g: a + scale * g, accum_grads, grads
                )
                is_last = loss_index == (
                    len(training_params.loss_calculation_times) - 1
                )
                if training_config.debug_training:

                    def grad_norm(grads):
                        return jnp.sqrt(
                            sum(
                                jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads)
                            )
                        )

                    jax.debug.print(
                        "grad norm at state {} = {} \n current total grads {}",
                        loss_index,
                        grad_norm(grads),
                        grad_norm(accum_grads),
                    )

                def apply_update(args):
                    network_params, opt_state, grads = args
                    updates, new_opt_state = optimizer.update(
                        grads, opt_state, network_params
                    )
                    new_net = eqx.apply_updates(network_params, updates)
                    new_grads = jax.tree_util.tree_map(
                        lambda x: jnp.zeros_like(x), grads
                    )
                    return new_net, new_opt_state, new_grads

                def skip_update(args):
                    return args

                network_params, opt_state, accum_grads = jax.lax.cond(
                    is_last,
                    apply_update,
                    skip_update,
                    (network_params, opt_state, accum_grads),
                )

            else:
                # Normal update after each loss
                updates, opt_state = optimizer.update(grads, opt_state, network_params)
                network_params = eqx.apply_updates(network_params, updates)
            # if config.return_snapshots or config.activate_snapshot_callback:
            #     time, primitive_state, snapshot_data = carry_loss
            #     carry = (
            #         time,
            #         primitive_state,
            #         snapshot_data,
            #         network_params,
            #         opt_state,
            #     )
            # else:
            #     time, primitive_state = carry_loss

            #     carry = time, primitive_state, network_params, opt_state

            if config.return_snapshots or config.activate_snapshot_callback:
                time, primitive_state, snapshot_data = carry_loss
                if training_config.accumulate_grads:
                    carry = (
                        time,
                        primitive_state,
                        snapshot_data,
                        network_params,
                        opt_state,
                        accum_grads,
                    )
                else:
                    carry = (
                        time,
                        primitive_state,
                        snapshot_data,
                        network_params,
                        opt_state,
                    )
            else:
                time, primitive_state = carry_loss
                if training_config.accumulate_grads:
                    carry = (
                        time,
                        primitive_state,
                        network_params,
                        opt_state,
                        accum_grads,
                    )
                else:
                    carry = (time, primitive_state, network_params, opt_state)

            return carry, jnp.array(loss_components.values())

    initial_network_params = params.corrector_params.network_params

    if config.return_snapshots or config.activate_snapshot_callback:
        if training_config.accumulate_grads:
            carry = (
                0.0,  # time
                primitive_state,
                snapshot_data,
                initial_network_params,
                opt_state,
                jax.tree_util.tree_map(
                    lambda x: jnp.zeros_like(x), initial_network_params
                ),
            )
        else:
            carry = (
                0.0,
                primitive_state,
                snapshot_data,
                initial_network_params,
                opt_state,
            )
    else:
        if training_config.accumulate_grads:
            carry = (
                0.0,
                primitive_state,
                initial_network_params,
                opt_state,
                jax.tree_util.tree_map(
                    lambda x: jnp.zeros_like(x), initial_network_params
                ),
            )
        else:
            carry = (
                0.0,
                primitive_state,
                initial_network_params,
                opt_state,
            )

    carry, losses = jax.lax.scan(
        train_step,
        carry,
        jnp.arange(len(training_params.loss_calculation_times)),
    )

    if config.return_snapshots or config.activate_snapshot_callback:
        if training_config.accumulate_grads:
            time, state, snapshot_data, network_params, opt_state, accum_grads = carry
        else:
            time, state, snapshot_data, network_params, opt_state = carry
    else:
        if training_config.accumulate_grads:
            time, state, network_params, opt_state, accum_grads = carry
        else:
            time, state, network_params, opt_state = carry

    if config.return_snapshots or config.activate_snapshot_callback:
        if config.return_snapshots:
            if config.snapshot_settings.return_final_state:
                snapshot_data = snapshot_data._replace(
                    final_state=_unpad(state, config)
                )
            return losses, network_params, opt_state, snapshot_data
        else:
            return losses, network_params, opt_state, _unpad(state, config)
    else:
        # unpad the primitive state if we padded it
        state = _unpad(state, config)

        return losses, network_params, opt_state, state
