import jax
import jax.numpy as jnp
from functools import partial

from jf1uids.CFL import cfl_time_step
from jf1uids.muscl_scheme import evolve_state

@partial(jax.jit, static_argnames=['config'])
def time_integration(primitive_state, config, params, helper_data):

    def update_step(carry):

        time, state = carry

        dt = cfl_time_step(state, params.dx, params.dt_max, params.gamma, params.C_cfl)

        state = evolve_state(state, params.dx, dt, params.gamma, config, params, helper_data)
        time += dt

        return (time, state)
    
    def condition(carry):
        t, _ = carry
        return t < params.t_end
    
    carry = (0.0, primitive_state)
    carry = jax.lax.while_loop(condition, update_step, carry)

    _, state = carry
    
    return state