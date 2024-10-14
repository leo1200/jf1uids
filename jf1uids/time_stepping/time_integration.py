import jax
import jax.numpy as jnp
from functools import partial

from equinox.internal._loop.checkpointed import checkpointed_while_loop

from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.option_classes.simulation_config import BACKWARDS, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams
from jf1uids.time_stepping._CFL import _cfl_time_step, _source_term_aware_time_step
from jf1uids.fluid_equations.fluid import calculate_total_energy, calculate_total_mass
from jf1uids._spatial_reconstruction.muscl_scheme import _evolve_state
from jf1uids._physics_modules.run_physics_modules import _run_physics_modules
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData

from timeit import default_timer as timer

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from typing import Union

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def time_integration(primitive_state: Float[Array, "num_vars num_cells"], config: SimulationConfig, params: SimulationParams, helper_data: HelperData) -> Union[Float[Array, "num_vars num_cells"], SnapshotData]:
    """Integrate the fluid equations in time. For the options of
    the time integration see the simulation configuration and
    the simulation parameters.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either the final state of the fluid
        after the time integration of snapshots of the time evolution.

    """

    # set dx appropriately
    config = config._replace(dx = config.box_size / (config.num_cells - 1))

    if config.fixed_timestep:
        return _time_integration_fixed_steps(primitive_state, config, params, helper_data)
    else:
        if config.differentiation_mode == BACKWARDS:
            return _time_integration_adaptive_backwards(primitive_state, config, params, helper_data)
        else:
            return _time_integration_adaptive_steps(primitive_state, config, params, helper_data)

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def _time_integration_fixed_steps(primitive_state: Float[Array, "num_vars num_cells"], config: SimulationConfig, params: SimulationParams, helper_data: HelperData) -> Union[Float[Array, "num_vars num_cells"], SnapshotData]:
    """ Fixed time stepping integration of the fluid equations.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either the final state of the fluid
        after the time integration of snapshots of the time evolution
    """

    if config.return_snapshots:
        raise NotImplementedError("return_snapshots only implemented with adaptive time stepping with forward mode option")

    dt = params.t_end / config.num_timesteps

    def update_step(_, state):

        state = _run_physics_modules(state, dt / 2, config, params, helper_data)
        state = _evolve_state(state, config.dx, dt, params.gamma, config, helper_data)
        state = _run_physics_modules(state, dt / 2, config, params, helper_data)

        return state
    
    # use lax fori_loop to unroll the loop
    state = jax.lax.fori_loop(0, config.num_timesteps, update_step, primitive_state)

    return state  

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def _time_integration_adaptive_steps(primitive_state: Float[Array, "num_vars num_cells"], config: SimulationConfig, params: SimulationParams, helper_data: HelperData) -> Union[Float[Array, "num_vars num_cells"], SnapshotData]:
    """Adaptive time stepping integration of the fluid equations.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either the final state of the fluid
        after the time integration of snapshots of the time evolution.
    """

    if config.return_snapshots:
        times = jnp.zeros(config.num_snapshots)
        states = jnp.zeros((config.num_snapshots, primitive_state.shape[0], primitive_state.shape[1]))
        total_mass = jnp.zeros(config.num_snapshots)
        total_energy = jnp.zeros(config.num_snapshots)
        current_checkpoint = 0
        snapshot_data = SnapshotData(times = times, states = states, total_mass = total_mass, total_energy = total_energy, current_checkpoint = current_checkpoint)

    def update_step(carry):

        if config.return_snapshots:
            time, state, snapshot_data = carry

            def update_snapshot_data(snapshot_data):
                times = snapshot_data.times.at[snapshot_data.current_checkpoint].set(time)
                states = snapshot_data.states.at[snapshot_data.current_checkpoint].set(state)
                total_mass = snapshot_data.total_mass.at[snapshot_data.current_checkpoint].set(calculate_total_mass(state, helper_data, config.num_ghost_cells))
                total_energy = snapshot_data.total_energy.at[snapshot_data.current_checkpoint].set(calculate_total_energy(state, helper_data, params.gamma, config.num_ghost_cells))
                current_checkpoint = snapshot_data.current_checkpoint + 1
                snapshot_data = snapshot_data._replace(times = times, states = states, total_mass = total_mass, total_energy = total_energy, current_checkpoint = current_checkpoint)
                return snapshot_data
            
            def dont_update_snapshot_data(snapshot_data):
                return snapshot_data

            snapshot_data = jax.lax.cond(time >= snapshot_data.current_checkpoint * params.t_end / config.num_snapshots, update_snapshot_data, dont_update_snapshot_data, snapshot_data)

            num_iterations = snapshot_data.num_iterations + 1
            snapshot_data = snapshot_data._replace(num_iterations = num_iterations)

        else:
            time, state = carry

        # dt = _cfl_time_step(state, config.dx, params.dt_max, params.gamma, params.C_cfl)

        # do not differentiate through the choice of the time step
        dt = jax.lax.stop_gradient(_source_term_aware_time_step(state, config, params, helper_data))

        # for now we mainly consider the stellar wind, a constant source term term, 
        # so the source is handled via a simple Euler step but generally 
        # a higher order method (in a split fashion) may be used

        state = _run_physics_modules(state, dt / 2, config, params, helper_data)
        state = _evolve_state(state, config.dx, dt, params.gamma, config, helper_data)
        state = _run_physics_modules(state, dt / 2, config, params, helper_data)

        time += dt

        if config.return_snapshots:
            carry = (time, state, snapshot_data)
        else:
            carry = (time, state)

        return carry
    
    def condition(carry):
        if config.return_snapshots:
            t, _, _ = carry
        else:
            t, _ = carry
        return t < params.t_end
    
    if config.return_snapshots:
        carry = (0.0, primitive_state, snapshot_data)
    else:
        carry = (0.0, primitive_state)
    
    start = timer()
    carry = jax.lax.while_loop(condition, update_step, carry)
    end = timer()
    duration = end - start

    if config.return_snapshots:
        _, state, snapshot_data = carry
        snapshot_data = snapshot_data._replace(runtime = duration)
        return snapshot_data
    else:
        _, state = carry
        return state
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def _time_integration_adaptive_backwards(primitive_state: Float[Array, "num_vars num_cells"], config: SimulationConfig, params: SimulationParams, helper_data: HelperData) -> Union[Float[Array, "num_vars num_cells"], SnapshotData]:
    """Adaptive time stepping integration of the fluid equations in backwards mode.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either the final state of the fluid
        after the time integration of snapshots of the time evolution.
    """

    if config.return_snapshots:
        raise NotImplementedError("return_snapshots only implemented with adaptive time stepping with forward mode option")

    def update_step(carry):

        time, state = carry

        # dt = _cfl_time_step(state, config.dx, params.dt_max, params.gamma, params.C_cfl)

        # do not differentiate through the choice of the time step
        dt = jax.lax.stop_gradient(_source_term_aware_time_step(state, config, params, helper_data))

        state = _run_physics_modules(state, dt / 2, config, params, helper_data)
        state = _evolve_state(state, config.dx, dt, params.gamma, config, helper_data)
        state = _run_physics_modules(state, dt / 2, config, params, helper_data)

        time += dt

        carry = (time, state)

        return carry
    
    def condition(carry):
        t, _ = carry
        return t < params.t_end
    
    carry = (0.0, primitive_state)
    
    carry = checkpointed_while_loop(condition, update_step, carry, checkpoints=config.num_checkpoints)

    _, state = carry
    return state