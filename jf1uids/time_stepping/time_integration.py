# general
from types import NoneType
import jax
import jax.numpy as jnp
from functools import partial

from equinox.internal._loop.checkpointed import checkpointed_while_loop

# type checking
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from typing import Union

# runtime debugging
from jax.experimental import checkify

# jf1uids constants
from jf1uids._geometry.boundaries import _boundary_handler
from jf1uids._geometry.geometry import _center_of_volume, _r_hat_alpha
from jf1uids.option_classes.simulation_config import BACKWARDS, CARTESIAN, CYLINDRICAL, FORWARDS, SPHERICAL, STATE_TYPE

# jf1uids containers
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.data_classes.simulation_helper_data import HelperData, get_helper_data
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData

# jf1uids functions
from jf1uids._state_evolution.evolve_state import _evolve_state
from jf1uids._physics_modules.run_physics_modules import _run_physics_modules
from jf1uids.time_stepping._timestep_estimator import _cfl_time_step, _source_term_aware_time_step
from jf1uids.fluid_equations.total_quantities import calculate_internal_energy, calculate_radial_momentum, calculate_total_mass
from jf1uids.fluid_equations.total_quantities import calculate_total_energy, calculate_kinetic_energy, calculate_gravitational_energy

# progress bar
from jf1uids.time_stepping._progress_bar import _show_progress

# timing
from timeit import default_timer as timer

from jf1uids.time_stepping._utils import _pad, _unpad

@jaxtyped(typechecker=typechecker)
def time_integration(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    snapshot_callable = None,
    sharding: Union[NoneType, jax.NamedSharding] = None
) -> Union[STATE_TYPE, SnapshotData]:
    
    """Integrate the fluid equations in time. For the options of
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

    helper_data_pad = get_helper_data(config, sharding, padded = True)

    if config.runtime_debugging:
        
        errors = checkify.user_checks | checkify.index_checks | checkify.float_checks | checkify.nan_checks | checkify.div_checks
        checked_integration = checkify.checkify(_time_integration, errors)

        err, final_state = checked_integration(primitive_state, config, params, helper_data, helper_data_pad, registered_variables, snapshot_callable)
        err.throw()
    
    else:

        if config.memory_analysis:
            compiled_step = _time_integration.lower(primitive_state, config, params, helper_data, helper_data_pad, registered_variables, snapshot_callable).compile()
            compiled_stats = compiled_step.memory_analysis()
            if compiled_stats is not None:
                # Calculate total memory usage including temporary storage, arguments, and outputs
                # Subtract alias size to avoid double-counting memory shared between different components
                total = compiled_stats.temp_size_in_bytes + compiled_stats.argument_size_in_bytes \
                    + compiled_stats.output_size_in_bytes - compiled_stats.alias_size_in_bytes
                print("=== Compiled time integration memory usage PER DEVICE ===")
                print(f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**2):.2f} MB")
                print(f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**2):.2f} MB")
                print(f"Total size: {total/(1024**2):.2f} MB")

        final_state = _time_integration(primitive_state, config, params, helper_data, helper_data_pad, registered_variables, snapshot_callable)

    return final_state

@partial(jax.jit, static_argnames=['config', 'registered_variables', 'snapshot_callable'])
@jaxtyped(typechecker=typechecker)
def _time_integration(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    helper_data_pad: HelperData,
    registered_variables: RegisteredVariables,
    snapshot_callable = None
) -> Union[STATE_TYPE, SnapshotData]:
    
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
        primitive_state = primitive_state.at[:-3, ...].set(_boundary_handler(primitive_state[:-3, ...], config))
    else:
        primitive_state = _boundary_handler(primitive_state, config)

    if config.return_snapshots or config.return_statistics:
        time_points = jnp.zeros(config.num_snapshots)

        if config.return_snapshots:
            states = jnp.zeros((config.num_snapshots, *original_shape))
        else:
            states = None

        if config.return_statistics:
            total_mass = jnp.zeros(config.num_snapshots)
            total_energy = jnp.zeros(config.num_snapshots)
            internal_energy = jnp.zeros(config.num_snapshots)
            kinetic_energy = jnp.zeros(config.num_snapshots)
            radial_momentum = jnp.zeros(config.num_snapshots)

        if config.self_gravity:
            gravitational_energy = jnp.zeros(config.num_snapshots)
        else:
            gravitational_energy = None

        current_checkpoint = 0

        snapshot_data = SnapshotData(time_points = time_points, states = states, total_mass = total_mass, total_energy = total_energy, internal_energy = internal_energy, kinetic_energy = kinetic_energy, gravitational_energy = gravitational_energy, current_checkpoint = current_checkpoint, radial_momentum = radial_momentum, final_state = None)

    elif config.activate_snapshot_callback:
        current_checkpoint = 0
        snapshot_data = SnapshotData(time_points = None, states = None, total_mass = None, total_energy = None, current_checkpoint = current_checkpoint)

    def update_step(carry):

        if config.return_snapshots or config.return_statistics:
            time, state, snapshot_data = carry

            def update_snapshot_data(time, state, snapshot_data):
                time_points = snapshot_data.time_points.at[snapshot_data.current_checkpoint].set(time)

                if config.return_snapshots:
                    # get the unpadded state to store in the snapshot
                    unpad_state = _unpad(state, config)
                    states = snapshot_data.states.at[snapshot_data.current_checkpoint].set(unpad_state)
                else:
                    states = None

                if config.return_statistics:
                    unpad_state = _unpad(state, config)
                    total_mass = snapshot_data.total_mass.at[snapshot_data.current_checkpoint].set(calculate_total_mass(unpad_state, helper_data, config))
                    total_energy = snapshot_data.total_energy.at[snapshot_data.current_checkpoint].set(calculate_total_energy(unpad_state, helper_data, params.gamma, params.gravitational_constant, config, registered_variables))

                    internal_energy = snapshot_data.internal_energy.at[snapshot_data.current_checkpoint].set(calculate_internal_energy(unpad_state, helper_data, params.gamma, config, registered_variables))
                    kinetic_energy = snapshot_data.kinetic_energy.at[snapshot_data.current_checkpoint].set(calculate_kinetic_energy(unpad_state, helper_data, config, registered_variables))

                    radial_momentum = snapshot_data.radial_momentum.at[snapshot_data.current_checkpoint].set(calculate_radial_momentum(unpad_state, helper_data, config, registered_variables))

                    if config.self_gravity:
                        gravitational_energy = snapshot_data.gravitational_energy.at[snapshot_data.current_checkpoint].set(calculate_gravitational_energy(unpad_state, helper_data, params.gravitational_constant, config, registered_variables))
                    else:
                        gravitational_energy = None
                else:
                    total_mass = None
                    total_energy = None
                    internal_energy = None
                    kinetic_energy = None
                    gravitational_energy = None
                    radial_momentum = None

                current_checkpoint = snapshot_data.current_checkpoint + 1
                snapshot_data = snapshot_data._replace(time_points = time_points, states = states, current_checkpoint = current_checkpoint, total_mass = total_mass, total_energy = total_energy, internal_energy = internal_energy, kinetic_energy = kinetic_energy, gravitational_energy = gravitational_energy, radial_momentum = radial_momentum)
                return snapshot_data
            
            def dont_update_snapshot_data(time, state, snapshot_data):
                return snapshot_data

            if config.use_specific_snapshot_timepoints:
                snapshot_data = jax.lax.cond(
                    jnp.abs(time - params.snapshot_timepoints[snapshot_data.current_checkpoint]) < 1e-12,
                    update_snapshot_data,
                    dont_update_snapshot_data,
                    time, state, snapshot_data
                )
            else:
                snapshot_data = jax.lax.cond(
                    time >= snapshot_data.current_checkpoint * params.t_end / config.num_snapshots,
                    update_snapshot_data,
                    dont_update_snapshot_data,
                    time, state, snapshot_data
                )

            num_iterations = snapshot_data.num_iterations + 1
            snapshot_data = snapshot_data._replace(num_iterations = num_iterations)

        elif config.activate_snapshot_callback:
            time, state, snapshot_data = carry

            def update_snapshot_data(snapshot_data):
                current_checkpoint = snapshot_data.current_checkpoint + 1
                snapshot_data = snapshot_data._replace(current_checkpoint = current_checkpoint)

                jax.debug.callback(snapshot_callable, time, state, registered_variables)

                return snapshot_data
            
            def dont_update_snapshot_data(snapshot_data):
                return snapshot_data

            snapshot_data = jax.lax.cond(time >= snapshot_data.current_checkpoint * params.t_end / config.num_snapshots, update_snapshot_data, dont_update_snapshot_data, snapshot_data)

            num_iterations = snapshot_data.num_iterations + 1
            snapshot_data = snapshot_data._replace(num_iterations = num_iterations)
        else:
            time, state = carry

        # dt = _cfl_time_step(state, config.grid_spacing, params.dt_max, params.gamma, params.C_cfl)

        # do not differentiate through the choice of the time step
        if not config.fixed_timestep:
            if config.source_term_aware_timestep:
                dt = jax.lax.stop_gradient(_source_term_aware_time_step(state, config, params, helper_data_pad, registered_variables, time))
            else:
                dt = jax.lax.stop_gradient(_cfl_time_step(state, config.grid_spacing, params.dt_max, params.gamma, config, registered_variables, params.C_cfl))
        else:
            dt = params.t_end / config.num_timesteps

        if config.use_specific_snapshot_timepoints:
            dt = jnp.minimum(dt, params.snapshot_timepoints[snapshot_data.current_checkpoint] - time)

        if config.exact_end_time and not config.use_specific_snapshot_timepoints:
            dt = jnp.minimum(dt, params.t_end - time)

        # for now we mainly consider the stellar wind, a constant source term term, 
        # so the source is handled via a simple Euler step but generally 
        # a higher order method (in a split fashion) may be used

        # state = _run_physics_modules(state, dt / 2, config, params, helper_data, registered_variables, time)
        state = _run_physics_modules(state, dt, config, params, helper_data_pad, registered_variables, time + dt)
        state = _evolve_state(state, dt, params.gamma, params.gravitational_constant, config, params, helper_data_pad, registered_variables)

        time += dt

        if config.use_specific_snapshot_timepoints:
            snapshot_data = jax.lax.cond(
                jnp.abs(time - params.t_end) < 1e-12,
                update_snapshot_data,
                dont_update_snapshot_data,
                time, state, snapshot_data
            )

        if config.progress_bar:
            jax.debug.callback(_show_progress, time, params.t_end)

        if config.return_snapshots or config.return_statistics or config.activate_snapshot_callback:
            carry = (time, state, snapshot_data)
        else:
            carry = (time, state)

        return carry
    
    def update_step_for(_, carry):
        return update_step(carry)
    
    def condition(carry):
        if config.return_snapshots or config.return_statistics or config.activate_snapshot_callback:
            t, _, _ = carry
        else:
            t, _ = carry
        return t < params.t_end
    
    if config.return_snapshots or config.return_statistics or config.activate_snapshot_callback:
        carry = (0.0, primitive_state, snapshot_data)
    else:
        carry = (0.0, primitive_state)
    
    if not config.fixed_timestep:
        if config.differentiation_mode == BACKWARDS:
            carry = checkpointed_while_loop(condition, update_step, carry, checkpoints = config.num_checkpoints)
        elif config.differentiation_mode == FORWARDS:
            carry = jax.lax.while_loop(condition, update_step, carry)
        else:
            raise ValueError("Unknown differentiation mode.")
    else:
        carry = jax.lax.fori_loop(0, config.num_timesteps, update_step_for, carry)


    if config.return_snapshots or config.return_statistics or config.activate_snapshot_callback:
        _, state, snapshot_data = carry

        if config.return_snapshots or config.return_statistics:
            snapshot_data = snapshot_data._replace(final_state = _unpad(state, config))
            return snapshot_data
        else:
            return state
    else:
        _, state = carry

        # unpad the primitive state if we padded it
        state = _unpad(state, config)

        return state