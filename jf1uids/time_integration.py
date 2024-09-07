import jax
import jax.numpy as jnp
from functools import partial

from jf1uids.CFL import cfl_time_step
from jf1uids.fluid import calculate_total_energy_proxy, calculate_total_mass_proxy
from jf1uids.muscl_scheme import evolve_state
from jf1uids.physics_modules.physical_sources import add_physical_sources
from jf1uids.simulation_checkpoint_data import CheckpointData

@partial(jax.jit, static_argnames=['config'])
def time_integration(primitive_state, config, params, helper_data):
    if config.fixed_timestep:
        return time_integration_fixed_steps(primitive_state, config, params, helper_data)
    else:
        return time_integration_adaptive_steps(primitive_state, config, params, helper_data)

@partial(jax.jit, static_argnames=['config'])
def time_integration_fixed_steps(primitive_state, config, params, helper_data):

    if config.checkpointing:
        raise NotImplementedError("Checkpointing not implemented")

    dt = params.t_end / config.num_timesteps

    def update_step(_, state):

        state = add_physical_sources(state, dt / 2, config, params, helper_data)
        state = evolve_state(state, params.dx, dt, params.gamma, config, params, helper_data)
        state = add_physical_sources(state, dt / 2, config, params, helper_data)

        return state
    
    # use lax fori_loop to unroll the loop
    state = jax.lax.fori_loop(0, config.num_timesteps, update_step, primitive_state)

    return state  


@partial(jax.jit, static_argnames=['config'])
def time_integration_adaptive_steps(primitive_state, config, params, helper_data):

    if config.checkpointing:
        times = jnp.zeros(config.num_checkpoints)
        states = jnp.zeros((config.num_checkpoints, primitive_state.shape[0], primitive_state.shape[1]))
        total_mass_proxy = jnp.zeros(config.num_checkpoints)
        total_energy_proxy = jnp.zeros(config.num_checkpoints)
        current_checkpoint = 0
        checkpoint_data = CheckpointData(times = times, states = states, total_mass_proxy = total_mass_proxy, total_energy_proxy = total_energy_proxy, current_checkpoint = current_checkpoint)

    def update_step(carry):

        if config.checkpointing:
            time, state, checkpoint_data = carry

            def update_checkpoint_data(checkpoint_data):
                times = checkpoint_data.times.at[checkpoint_data.current_checkpoint].set(time)
                states = checkpoint_data.states.at[checkpoint_data.current_checkpoint].set(state)
                total_mass_proxy = checkpoint_data.total_mass_proxy.at[checkpoint_data.current_checkpoint].set(calculate_total_mass_proxy(state, helper_data, params))
                total_energy_proxy = checkpoint_data.total_energy_proxy.at[checkpoint_data.current_checkpoint].set(calculate_total_energy_proxy(state, helper_data, params))
                current_checkpoint = checkpoint_data.current_checkpoint + 1
                checkpoint_data = checkpoint_data._replace(times = times, states = states, total_mass_proxy = total_mass_proxy, total_energy_proxy = total_energy_proxy, current_checkpoint = current_checkpoint)
                return checkpoint_data
            
            def dont_update_checkpoint_data(checkpoint_data):
                return checkpoint_data

            checkpoint_data = jax.lax.cond(time >= checkpoint_data.current_checkpoint * params.t_end / config.num_checkpoints, update_checkpoint_data, dont_update_checkpoint_data, checkpoint_data)

        else:
            time, state = carry

        dt = cfl_time_step(state, params.dx, params.dt_max, params.gamma, params.C_cfl)

        state = add_physical_sources(state, dt / 2, config, params, helper_data)
        state = evolve_state(state, params.dx, dt, params.gamma, config, params, helper_data)
        state = add_physical_sources(state, dt / 2, config, params, helper_data)

        time += dt

        if config.checkpointing:
            carry = (time, state, checkpoint_data)
        else:
            carry = (time, state)

        return carry
    
    def condition(carry):
        if config.checkpointing:
            t, _, _ = carry
        else:
            t, _ = carry
        return t < params.t_end
    
    if config.checkpointing:
        carry = (0.0, primitive_state, checkpoint_data)
    else:
        carry = (0.0, primitive_state)
    
    carry = jax.lax.while_loop(condition, update_step, carry)

    if config.checkpointing:
        _, state, checkpoint_data = carry
        return checkpoint_data
    else:
        _, state = carry
        return state