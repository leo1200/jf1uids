import jax
import jax.numpy as jnp
from functools import partial

from equinox.internal._loop.checkpointed import checkpointed_while_loop

from jf1uids.option_classes.simulation_config import BACKWARDS
from jf1uids.time_stepping.CFL import _cfl_time_step
from jf1uids.fluid_equations.fluid import calculate_total_energy, calculate_total_mass
from jf1uids.spatial_reconstruction.muscl_scheme import evolve_state
from jf1uids.physics_modules.run_physics_modules import run_physics_modules
from jf1uids.data_classes.simulation_checkpoint_data import CheckpointData

from timeit import default_timer as timer

@partial(jax.jit, static_argnames=['config'])
def _source_term_aware_time_step(state, config, params, helper_data):
    """
    Calculate the time step based on the CFL condition and the source terms
    """

    # calculate the time step based on the CFL condition
    dt = _cfl_time_step(state, config.dx, params.dt_max, params.gamma, params.C_cfl)

    # == experimental: correct the CFL time step based on the physical sources ==
    hypothetical_new_state = run_physics_modules(state, dt, config, params, helper_data)
    dt = _cfl_time_step(hypothetical_new_state, config.dx, params.dt_max, params.gamma, params.C_cfl)
    # ===========================================================================

    return dt

@partial(jax.jit, static_argnames=['config'])
def time_integration(primitive_state, config, params, helper_data):

    # set dx appropriately
    config = config._replace(dx = config.box_size / (config.num_cells - 1))

    if config.fixed_timestep:
        return time_integration_fixed_steps(primitive_state, config, params, helper_data)
    else:
        if config.differentiation_mode == BACKWARDS:
            return time_integration_adaptive_backwards(primitive_state, config, params, helper_data)
        else:
            return time_integration_adaptive_steps(primitive_state, config, params, helper_data)

@partial(jax.jit, static_argnames=['config'])
def time_integration_fixed_steps(primitive_state, config, params, helper_data):

    if config.intermediate_saves:
        raise NotImplementedError("intermediate_saves only implemented with adaptive time stepping with forward mode option")

    dt = params.t_end / config.num_timesteps

    def update_step(_, state):

        state = run_physics_modules(state, dt / 2, config, params, helper_data)
        state = evolve_state(state, config.dx, dt, params.gamma, config, helper_data)
        state = run_physics_modules(state, dt / 2, config, params, helper_data)

        return state
    
    # use lax fori_loop to unroll the loop
    state = jax.lax.fori_loop(0, config.num_timesteps, update_step, primitive_state)

    return state  


@partial(jax.jit, static_argnames=['config'])
def time_integration_adaptive_steps(primitive_state, config, params, helper_data):

    if config.intermediate_saves:
        times = jnp.zeros(config.num_saves)
        states = jnp.zeros((config.num_saves, primitive_state.shape[0], primitive_state.shape[1]))
        total_mass_proxy = jnp.zeros(config.num_saves)
        total_energy_proxy = jnp.zeros(config.num_saves)
        current_checkpoint = 0
        checkpoint_data = CheckpointData(times = times, states = states, total_mass_proxy = total_mass_proxy, total_energy_proxy = total_energy_proxy, current_checkpoint = current_checkpoint)

    def update_step(carry):

        if config.intermediate_saves:
            time, state, checkpoint_data = carry

            def update_checkpoint_data(checkpoint_data):
                times = checkpoint_data.times.at[checkpoint_data.current_checkpoint].set(time)
                states = checkpoint_data.states.at[checkpoint_data.current_checkpoint].set(state)
                total_mass_proxy = checkpoint_data.total_mass_proxy.at[checkpoint_data.current_checkpoint].set(calculate_total_mass(state, helper_data, config.dx, config.num_ghost_cells))
                total_energy_proxy = checkpoint_data.total_energy_proxy.at[checkpoint_data.current_checkpoint].set(calculate_total_energy(state, helper_data, config.dx, params.gamma, config.num_ghost_cells))
                current_checkpoint = checkpoint_data.current_checkpoint + 1
                checkpoint_data = checkpoint_data._replace(times = times, states = states, total_mass_proxy = total_mass_proxy, total_energy_proxy = total_energy_proxy, current_checkpoint = current_checkpoint)
                return checkpoint_data
            
            def dont_update_checkpoint_data(checkpoint_data):
                return checkpoint_data

            checkpoint_data = jax.lax.cond(time >= checkpoint_data.current_checkpoint * params.t_end / config.num_saves, update_checkpoint_data, dont_update_checkpoint_data, checkpoint_data)

            num_iterations = checkpoint_data.num_iterations + 1
            checkpoint_data = checkpoint_data._replace(num_iterations = num_iterations)

        else:
            time, state = carry

        # dt = _cfl_time_step(state, config.dx, params.dt_max, params.gamma, params.C_cfl)

        # do not differentiate through the choice of the time step
        dt = jax.lax.stop_gradient(_source_term_aware_time_step(state, config, params, helper_data))

        # for now we mainly consider the stellar wind, a constant source term term, 
        # so the source is handled via a simple Euler step but generally 
        # a higher order method (in a split fashion) may be used

        state = run_physics_modules(state, dt / 2, config, params, helper_data)
        state = evolve_state(state, config.dx, dt, params.gamma, config, helper_data)
        state = run_physics_modules(state, dt / 2, config, params, helper_data)

        time += dt

        if config.intermediate_saves:
            carry = (time, state, checkpoint_data)
        else:
            carry = (time, state)

        return carry
    
    def condition(carry):
        if config.intermediate_saves:
            t, _, _ = carry
        else:
            t, _ = carry
        return t < params.t_end
    
    if config.intermediate_saves:
        carry = (0.0, primitive_state, checkpoint_data)
    else:
        carry = (0.0, primitive_state)
    
    start = timer()
    carry = jax.lax.while_loop(condition, update_step, carry)
    end = timer()
    duration = end - start

    if config.intermediate_saves:
        _, state, checkpoint_data = carry
        checkpoint_data = checkpoint_data._replace(runtime = duration)
        return checkpoint_data
    else:
        _, state = carry
        return state
    

@partial(jax.jit, static_argnames=['config'])
def time_integration_adaptive_backwards(primitive_state, config, params, helper_data):

    def update_step(carry):

        time, state = carry

        # dt = _cfl_time_step(state, config.dx, params.dt_max, params.gamma, params.C_cfl)

        # do not differentiate through the choice of the time step
        dt = jax.lax.stop_gradient(_source_term_aware_time_step(state, config, params, helper_data))

        state = run_physics_modules(state, dt / 2, config, params, helper_data)
        state = evolve_state(state, config.dx, dt, params.gamma, config, helper_data)
        state = run_physics_modules(state, dt / 2, config, params, helper_data)

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