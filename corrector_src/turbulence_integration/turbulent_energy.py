import jax
from functools import partial
import jax.numpy as jnp
from jf1uids.fluid_equations.fluid import (
    get_absolute_velocity,
    total_energy_from_primitives,
)
from jf1uids._physics_modules._mhd._vector_maths import (
    curl2D,
    curl3D,
    divergence3D,
    divergence2D,
)
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def calculate_turbulent_energy(state, registered_variables, config):
    """
    Turbulent energy from 10.48550/arXiv.2410.23339 eqn 8, only makes sense in dns regimes
    """
    rho = state[registered_variables.density_index]
    u = get_absolute_velocity(state, config, registered_variables)

    u_n = jnp.sum(rho * u) / jnp.sum(rho)
    e_turb = (jnp.sum(rho * u**2) - jnp.sum(rho) * jnp.sum(u_n**2)) / 2
    return e_turb


def adiabatic_turbulent_energy(state, registered_variables, config, dt):
    vx = state[registered_variables.velocity_index[0]]
    vy = state[registered_variables.velocity_index[1]]
    vz = state[registered_variables.velocity_index[2]]
    v = jnp.stack([vx, vy, vz], axis=0)
    e_turb = state[registered_variables.e_turb_index]

    gamma_turb = 5 / 3
    P_turb = (gamma_turb - 1.0) * e_turb

    if config.dimensionality == 3:
        div_v = divergence3D(v, config.grid_spacing)
        adv_flux = divergence3D(e_turb[None, ...] * v, config.grid_spacing)

    elif config.dimensionality == 2:
        div_v = divergence2D(v, config.grid_spacing)
        adv_flux = divergence2D(e_turb[None, ...] * v, config.grid_spacing)

    # Adiabatic update
    e_turb_new = e_turb - dt * (adv_flux + P_turb * div_v)

    # Positivity clamp
    e_turb_new = jnp.maximum(e_turb_new, 0.0)

    return e_turb_new


def _evolve_turb_energy(state, dt, gamma, config, registered_variables):
    # e_turb_adi = adiabatic_turbulent_energy(
    #     state=state, registered_variables=registered_variables, config=config, dt=dt
    # )
    # e_turb_adi = jnp.maximum(e_turb_adi, 0.0)
    rho = state[registered_variables.density_index]

    u = get_absolute_velocity(state, config, registered_variables)
    p = state[registered_variables.pressure_index]

    E = total_energy_from_primitives(rho, u, p, gamma)
    # jax.debug.print(
    #     "negative values found in e_turb {E}",
    #     E=jnp.any(jnp.isnan(E)),
    # )

    e_turb = E - (rho * u) ** 2 / (2 * rho) - p / (gamma - 1)

    # jax.debug.print(
    #     "negative values found in e_turb {e_turb}",
    #     e_turb=jnp.any(jnp.isnan(e_turb)),
    # )

    # subgrid turbulence dissipation
    C_epsilon = 1.6
    jax.debug.print(
        "negative found in e_turb {neg_e_turb}", neg_e_turb=jnp.any(e_turb < 0)
    )
    epsilon = (
        C_epsilon * (e_turb) ** (3 / 2) / (config.grid_spacing * rho ** (1 / 2) + 1e-10)
    )
    jax.debug.print(
        "nan found in epsilon numerator {epsilon}, divisor {divisor}",
        epsilon=jnp.any(jnp.isnan((e_turb) ** (3 / 2))),
        divisor=jnp.any(jnp.isnan(rho ** (1 / 2))),
    )

    e_turb = (
        e_turb - dt * epsilon
    )  # missing here two terms, according to seminov the turb diffusion is rather weak so i leave it for later
    # jax.debug.print(
    #     "negative values found in e_turb_after {e_turb}",
    #     e_turb=jnp.any(jnp.isnan(e_turb)),
    # )

    f = C_epsilon * (dt / config.grid_spacing) * jnp.sqrt(e_turb / rho)
    delta_e_turb = e_turb * (f * (f + 4)) / (f + 2) ** 2

    e_turb_new = e_turb - delta_e_turb
    e_turb_new = jnp.maximum(e_turb_new, 0.0)

    e_th_primitive = p / (gamma - 1.0)
    e_th_new = e_th_primitive + delta_e_turb

    # convert back to pressure primitive
    p_new = (gamma - 1.0) * e_th_new
    state = state.at[registered_variables.e_turb_index].set(e_turb_new)
    state = state.at[registered_variables.pressure_index].set(p_new)
    # jax.debug.print(
    #     "negative values found in delta_e_turb {delta_e_turb_neg}",
    #     delta_e_turb_neg=jnp.any(delta_e_turb < 0),
    # )
    # jax.debug.print(
    #     "negative values found in e_th_new {e_th_new}",
    #     e_th_new=jnp.any(e_th_new < 0),
    # )
    # jax.debug.print(
    #     "negative values found in e_turb {e_turb}",
    #     e_turb=jnp.any(jnp.isnan(e_turb)),
    # )

    return state


def initialize_e_turb(state, registered_variables, fraction=0.01):
    """
    Initialize e_turb as a fraction of thermal energy.
    """
    # p_th = state[registered_variables.pressure_index]
    # gamma = 5 / 3  # or pass from params

    # e_th = p_th / (gamma - 1.0)
    # e_turb_init = fraction * e_th
    # state = state.at[registered_variables.e_turb_index].set(e_turb_init)
    # # optionally subtract from thermal energy to conserve total E
    # e_th_new = e_th - e_turb_init
    # p_th_new = (gamma - 1.0) * e_th_new
    e_turb_init = jnp.zeros_like(state[registered_variables.density_index])
    state = state.at[registered_variables.e_turb_index].set(e_turb_init)
    return state
