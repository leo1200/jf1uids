# numerics
from helpers import calculate_internal_energy_from_energy, calculate_pressure_from_internal_energy, calculate_speed_of_sound, calculate_total_energy_from_primitive_vars
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

jax.config.update("jax_debug_nans", True)

# plotting
from astropy.visualization import quantity_support
quantity_support()
import matplotlib.pyplot as plt

from unit_helpers import CodeUnits
from astropy import units as u
import astropy.constants as c
from weaver import Weaver
from astropy.constants import m_p

# for now using CPU as of outdated NVIDIA Driver
jax.config.update('jax_platform_name', 'cpu')

def get_initial_state(rho, u, p, gamma):
    """
    Get initial state.
    
        Parameters
        ----------
        rho : float
        Density.
        u : float
        Velocity.
        p : float
        Pressure.
    
        Returns
        -------
        Initial state tensor (rho, m = rho * u, E).
    """
    m = rho * u
    E = calculate_total_energy_from_primitive_vars(rho, u, p, gamma)
    return jnp.stack([rho, m, E], axis=0)

def geometric_source_term(state, r, alpha, gamma):
    """
    Geometric source term.
    
        Parameters
        ----------
        state : tensor
        State tensor (rho, m = rho * u, E).
        r : tensor
        Radial coordinate.
        alpha : float
        Geometry parameter.
    
        Returns
        -------
        Source term tensor.
    """
    rho = state[0]
    m = state[1]
    E = state[2]
    u = m / rho
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), rho, gamma)
    # S = -alpha / r * [rho u, rho u^2, u (E + p)]
    return -alpha / r * jnp.stack([rho * u, rho * u**2, u * (E + p)], axis=0)

def get_center_flux(state, gamma):
    """
    Get flux tensor.
    
        Parameters
        ----------
        state : tensor
        State tensor (rho, m = rho * u, E).
    
        Returns
        -------
        Flux tensor (rho * u, rho * u^2 + p, u * (E + p)).
    """
    rho = state[0]
    m = state[1]
    E = state[2]
    u = m / rho
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), rho, gamma)
    return jnp.stack([m, m * u + p, u * (E + p)], axis=0)

def get_cfl_time_step(states, gamma, delta_r, C_cfl, dt_max):
    rho = states[0]
    u = states[1] / states[0]
    E = calculate_total_energy_from_primitive_vars(states[0], states[1], states[2], gamma)
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), rho, gamma)
    sound_speeds = calculate_speed_of_sound(states[0], p, gamma)

    # get maximum local speeds
    max_local_speeds = jnp.max(jnp.maximum(jnp.abs(u + sound_speeds), jnp.abs(u - sound_speeds)))

    # get the time step
    dt = C_cfl * delta_r / max_local_speeds

    dt = jnp.minimum(dt, dt_max)

    return dt


def lax_friedrich_local_split(fluxes, states, gamma):
    rho = states[0]
    u = states[1] / states[0]
    E = calculate_total_energy_from_primitive_vars(states[0], states[1], states[2], gamma)
    p = calculate_pressure_from_internal_energy(calculate_internal_energy_from_energy(E, rho, u), rho, gamma)
    sound_speeds = calculate_speed_of_sound(states[0], p, gamma)

    # get maximum local speeds
    max_local_speeds = jnp.max(jnp.maximum(jnp.abs(u + sound_speeds), jnp.abs(u - sound_speeds)))

    # print("max_local_speeds: ", max_local_speeds)

    # get f+ = 1/2 * (f + max_local_speeds * u), f- = 1/2 * (f - max_local_speeds * u)
    f_plus = 0.5 * (fluxes + max_local_speeds * states)
    f_minus = 0.5 * (fluxes - max_local_speeds * states)

    return f_plus, f_minus

def get_weno5_flux(f_plus, f_minus):

    # calculate the plus and minus fluxes

    # the plus fluxes are
    # f_0 = 1/3 f_{i-2} - 7/6 f_{i-1} + 11/6 f_i
    # f_1 = -1/6 f_{i-1} + 5/6 f_i + 1/3 f_{i+1}
    # f_2 = 1/3 f_i + 5/6 f_{i+1} - 1/6 f_{i+2}

    # the minus fluxes are
    # f_0 = 1/3 f_{i+3} - 7/6 f_{i+2} + 11/6 f_{i+1}
    # f_1 = -1/6 f_{i+2} + 5/6 f_{i+1} + 1/3 f_i
    # f_2 = 1/3 f_{i+1} + 5/6 f_i - 1/6 f_{i-1}

    # with this, we can calculate fluxes for the indices [2:-3] the +1/2 fluxes on
    # each index, such that for f_{i}, one uses f[2:-3], for f_{i+1}, one uses f[3:-2] etc.

    # f_i: f[2:-3]
    # f_{i+1}: f[3:-2]
    # f_{i+2}: f[4:-1]
    # f_{i+3}: f[5:]
    # f_{i-1}: f[1:-4]
    # f_{i-2}: f[:-5]

    # plus fluxes
    # f_0 = 1/3 f_{i-2} - 7/6 f_{i-1} + 11/6 f_i
    f_0_plus = 1/3 * f_plus[:, :-5] - 7/6 * f_plus[:, 1:-4] + 11/6 * f_plus[:, 2:-3]
    f_1_plus = -1/6 * f_plus[:, 1:-4] + 5/6 * f_plus[:, 2:-3] + 1/3 * f_plus[:, 3:-2]
    f_2_plus = 1/3 * f_plus[:, 2:-3] + 5/6 * f_plus[:, 3:-2] - 1/6 * f_plus[:, 4:-1]

    # minus fluxes
    # f_0 = 1/3 f_{i+3} - 7/6 f_{i+2} + 11/6 f_{i+1}
    f_0_minus = 1/3 * f_minus[:, 5:] - 7/6 * f_minus[:, 4:-1] + 11/6 * f_minus[:, 3:-2]
    f_1_minus = -1/6 * f_minus[:, 4:-1] + 5/6 * f_minus[:, 3:-2] + 1/3 * f_minus[:, 2:-3]
    f_2_minus = 1/3 * f_minus[:, 3:-2] + 5/6 * f_minus[:, 2:-3] - 1/6 * f_minus[:, 1:-4]

    # get the positive smoothness indicators
    # beta_0 = 13/12 (f_{i-2} - 2 f_{i-1} + f_i)^2 + 1/4 (f_{i-2} - 4 f_{i-1} + 3 f_i)^2
    # beta_1 = 13/12 (f_{i-1} - 2 f_i + f_{i+1})^2 + 1/4 (f_{i-1} - f_{i+1})^2
    # beta_2 = 13/12 (f_i - 2 f_{i+1} + f_{i+2})^2 + 1/4 (3 f_i - 4 f_{i+1} + f_{i+2})^2

    beta_0_plus = 13/12 * (f_plus[:, :-5] - 2 * f_plus[:, 1:-4] + f_plus[:, 2:-3])**2 + 1/4 * (f_plus[:, :-5] - 4 * f_plus[:, 1:-4] + 3 * f_plus[:, 2:-3])**2
    beta_1_plus = 13/12 * (f_plus[:, 1:-4] - 2 * f_plus[:, 2:-3] + f_plus[:, 3:-2])**2 + 1/4 * (f_plus[:, 1:-4] - f_plus[:, 3:-2])**2
    beta_2_plus = 13/12 * (f_plus[:, 2:-3] - 2 * f_plus[:, 3:-2] + f_plus[:, 4:-1])**2 + 1/4 * (3 * f_plus[:, 2:-3] - 4 * f_plus[:, 3:-2] + f_plus[:, 4:-1])**2
    
    tau_5_plus = jnp.abs(beta_2_plus - beta_0_plus)

    # get the negative smoothness indicators
    # beta_0 = 13/12 (f_{i+1} - 2 f_{i+2} + f_{i+3})^2 + 1/4 (3 f_{i+1} - 4 f_{i+2} + f_{i+3})^2
    # beta_1 = 13/12 (f_i - 2 f_{i+1} + f_{i+2})^2 + 1/4 (f_i - f_{i+2})^2
    # beta_2 = 13/12 (f_{i-1} - 2 f_i + f_{i+1})^2 + 1/4 (f_{i-1} - 4 f_i + 3 f_{i+1})^2

    beta_0_minus = 13/12 * (f_minus[:, 3:-2] - 2 * f_minus[:, 4:-1] + f_minus[:, 5:])**2 + 1/4 * (3 * f_minus[:, 3:-2] - 4 * f_minus[:, 4:-1] + f_minus[:, 5:])**2
    beta_1_minus = 13/12 * (f_minus[:, 2:-3] - 2 * f_minus[:, 3:-2] + f_minus[:, 4:-1])**2 + 1/4 * (f_minus[:, 2:-3] - f_minus[:, 4:-1])**2
    beta_2_minus = 13/12 * (f_minus[:, 1:-4] - 2 * f_minus[:, 2:-3] + f_minus[:, 3:-2])**2 + 1/4 * (f_minus[:, 1:-4] - 4 * f_minus[:, 2:-3] + 3 * f_minus[:, 3:-2])**2

    tau_5_minus = jnp.abs(beta_2_minus - beta_0_minus)

    # get the linear weights
    gamma_0 = 1/10
    gamma_1 = 3/5
    gamma_2 = 3/10

    q = 2

    # get the nonlinear weights
    epsilon = 1e-6
    omega_tilde_0_plus = gamma_0 * (1 + tau_5_plus / (beta_0_plus + epsilon)) ** q
    omega_tilde_1_plus = gamma_1 * (1 + tau_5_plus / (beta_1_plus + epsilon)) ** q
    omega_tilde_2_plus = gamma_2 * (1 + tau_5_plus / (beta_2_plus + epsilon)) ** q

    omega_tilde_0_minus = gamma_0 * (1 + tau_5_minus / (beta_0_minus + epsilon)) ** q
    omega_tilde_1_minus = gamma_1 * (1 + tau_5_minus / (beta_1_minus + epsilon)) ** q
    omega_tilde_2_minus = gamma_2 * (1 + tau_5_minus / (beta_2_minus + epsilon)) ** q

    # get the weights
    normalizing_factor_plus = omega_tilde_0_plus + omega_tilde_1_plus + omega_tilde_2_plus
    omega_0_plus = omega_tilde_0_plus / normalizing_factor_plus
    omega_1_plus = omega_tilde_1_plus / normalizing_factor_plus
    omega_2_plus = omega_tilde_2_plus / normalizing_factor_plus

    normalizing_factor_minus = omega_tilde_0_minus + omega_tilde_1_minus + omega_tilde_2_minus
    omega_0_minus = omega_tilde_0_minus / normalizing_factor_minus
    omega_1_minus = omega_tilde_1_minus / normalizing_factor_minus
    omega_2_minus = omega_tilde_2_minus / normalizing_factor_minus

    # get the WENO fluxes
    f_Weno_plus = omega_0_plus * f_0_plus + omega_1_plus * f_1_plus + omega_2_plus * f_2_plus
    f_Weno_minus = omega_0_minus * f_0_minus + omega_1_minus * f_1_minus + omega_2_minus * f_2_minus

    f_Weno = f_Weno_plus + f_Weno_minus

    return f_Weno

def state_time_deriv(states, r, delta_r, alpha, gamma):

    # get the fluxes
    fluxes = get_center_flux(states, gamma)

    # get the local Lax-Friedrichs fluxes
    f_plus, f_minus = lax_friedrich_local_split(fluxes, states, gamma)

    # get the WENO flux
    f_Weno = get_weno5_flux(f_plus, f_minus)

    # get the geometric source term
    S = geometric_source_term(states, r, alpha, gamma)

    wind_source_term = wind_source(r)

    # print f_Weno
    # print(f_Weno)

    # get the time derivative
    dU_dt = -(f_Weno[:, 1:] - f_Weno[:, :-1]) / delta_r + S[:, 3:-3]

    # add the wind source term
    dU_dt = dU_dt.at[0, 3:13].add(wind_source_term[0])
    dU_dt = dU_dt.at[2, 3:13].add(wind_source_term[2])

    # print the shape of dU_dt
    # print(dU_dt.shape)

    return dU_dt

def euler_stepper(state, r, delta_r, alpha, gamma, dt, T):

    t = 0

    progress = 0

    while t < T:

        dt = get_cfl_time_step(state, gamma, delta_r, 0.45)
            
        # get the time derivative
        dU_dt = state_time_deriv(state, r, delta_r, alpha, gamma)

        # update the state
        state = state.at[:, 3:-3].add(dt * dU_dt)

        # print the state
        # print("state: ", state)

        t += dt

        # if dt passess another percent, print progress
        progress_new = int(t / t_end * 100)
        if progress_new > progress:
            print(f"Progress: {progress_new}%")
            progress = progress_new

    return state

def rk3_stepper(state, r, delta_r, alpha, gamma, C_cfl, dt, T):

    t = 0

    progress = 0

    while t < T:

        dt = get_cfl_time_step(state, gamma, delta_r, C_cfl, dt)


        # inner reflective boundary condition
        # innermost cell
        state = state.at[:, 0].set(state[:, 3])
        state = state.at[1, 0].set(-state[1, 3])
        # second innermost cell
        state = state.at[:, 1].set(state[:, 3])
        state = state.at[1, 1].set(-state[1, 3])
        # third innermost cell
        state = state.at[:, 2].set(state[:, 3])
        state = state.at[1, 2].set(-state[1, 3])
        

        # outer outflow boundary condition
        # outermost cell
        state = state.at[:, -1].set(state[:, -4])
        # second outermost cell
        state = state.at[:, -2].set(state[:, -4])
        # third outermost cell
        state = state.at[:, -3].set(state[:, -4])

        # dt = get_cfl_time_step(state, gamma, delta_r, C_cfl, dt)
            
        # get the time derivative
        dU_dt = state_time_deriv(state, r, delta_r, alpha, gamma)

        # update the state
        u1 = state.at[:, 3:-3].add(dt * dU_dt)
        u2 = state.at[:, 3:-3].set(0.75 * state[:, 3:-3] + 0.25 * u1[:, 3:-3] + 0.25 * dt * state_time_deriv(u1, r, delta_r, alpha, gamma))
        state = state.at[:, 3:-3].set(1/3 * state[:, 3:-3] + 2/3 * u2[:, 3:-3] + 2/3 * dt * state_time_deriv(u2, r, delta_r, alpha, gamma))

        # print the state
        # print("state: ", state)

        t += dt

        # if dt passess another percent, print progress
        progress_new = int(t / T * 100)
        if progress_new > progress:
            print(f"Progress: {progress_new}%")
            progress = progress_new

    return state

# =============== EXAMPLE INITIAL CONDITIONS ===============

# # for testing purposes, setup sod shock tube problem
# N_grid = 400
# L = 1.0
# r = jnp.linspace(0, L, N_grid)
# alpha = 2 # 0 -> cartesian, 1 -> cylindrical, 2 -> spherical
# shock_pos = 0.5
# gamma = 5/3

# rho = jnp.where(r < shock_pos, 1.0, 0.125)
# u = jnp.where(r < shock_pos, 0, 0)
# p = jnp.where(r < shock_pos, 1, 0.1)

# # get initial state
# state = get_initial_state(rho, u, p, gamma)

# # simulation settings
# dx = L / N_grid

# dt = 0.0005
# t_end = 0.2

# # run the simulation
# state = rk3_stepper(state, r, dx, alpha, gamma, dt, t_end)

# # print the final state
# # print("Final state: ", state)

# # plot final state, density, velocity, pressure, internal energy
# rho = state[0]
# u = state[1] / state[0]
# E = state[2]
# internal_energy = calculate_internal_energy_from_energy(state[2], state[0], u)
# p = calculate_pressure_from_internal_energy(internal_energy, rho, gamma)
# entropic_function = p / state[0]**gamma

# fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# axs[0, 0].plot(r, rho)
# axs[0, 0].set_title("Density")

# axs[0, 1].plot(r, p)
# axs[0, 1].set_title("Pressure")

# axs[1, 0].plot(r, u)
# axs[1, 0].set_title("Velocity")

# #axs[1, 1].plot(r, internal_energy / rho)
# # use log scale
# axs[1, 1].plot(r, internal_energy)
# axs[1, 1].set_title("internal energy")

# plt.savefig("weno_test.png")

# =========================================================

# ================== Physics Modules ======================

def dummy_wind(state, A, B, num_cells):
    state = state.at[0, 1:num_cells].add(A)
    state = state.at[2, 1:num_cells].add(B)
    return state

def wind_source(r):

    # code units
    code_length = 3 * u.parsec
    code_mass = 1e5 * u.M_sun
    code_velocity = 1 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)


    # mass of the star
    M_star = 40 * u.M_sun

    num_cells = 10

    # wind features
    StellarWindFinalVelocity = 2000 * u.km / u.s
    StellarWindMassLossRate = 2.965e-3 / (1e6 * u.yr) * M_star

    m_dot = StellarWindMassLossRate.to(code_units.code_mass / code_units.code_time).value


    # mass injection
    r_inj = r[num_cells + 3]
    V = 4/3 * jnp.pi * r_inj**3
    rho_dot = m_dot / V
    # state = state.at[0, 3:num_cells].add(rho_dot)

    # energy injection
    E_dot = 0.5 * StellarWindFinalVelocity**2 * StellarWindMassLossRate
    E_dot = E_dot.to(code_units.code_mass * code_units.code_velocity**2 / code_units.code_time).value / V

    # state = state.at[2, 3:num_cells].add(E_dot)

    # # energy injection
    # momentum_dot = StellarWindFinalVelocity.to(code_units.code_velocity) * state[0, 1:num_cells] * dt

    # # inject momentum
    # state = state.at[1, 1:num_cells].add(momentum_dot)

    wind_source = jnp.array([rho_dot, 0, E_dot])

    return wind_source


# =========================================================

# =============== EXAMPLE INITIAL CONDITIONS ===============

# for testing purposes, setup sod shock tube problem
N_grid = 200
L = 1.0

dx = L / N_grid
alpha = 2 # 0 -> cartesian, 1 -> cylindrical, 2 -> spherical
shock_pos = 0.4
gamma = 5/3

# code units
code_length = 3 * u.parsec
code_mass = 1e5 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

r = jnp.linspace(dx / 2, L + 1/2 * dx + 6 * dx, N_grid + 6) * code_units.code_length

print(r)

# ISM density
rho_0 = 2 * c.m_p / u.cm**3
# ISM pressure
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

print(rho_0.to(code_units.code_density))

# wanted final time
t_final = 0.6 * 1e4 * u.yr
code_units.print_simulation_parameters(t_final)

rho_init = jnp.ones(N_grid + 6) * rho_0.to(code_units.code_density).value
u_init = jnp.zeros(N_grid + 6)
p_init = jnp.ones(N_grid + 6) * p_0.to(code_units.code_pressure).value

# get initial state
state = get_initial_state(rho_init, u_init, p_init, gamma)

# simulation settings
C_CFL = 0.4

t_end = t_final.to(code_units.code_time).value
dt_max = 0.0001 * t_end

# print end time
print(f"End time: {t_end}")

# ========== WIND ===========


# ===========================

# run the simulation
C_cfl = 0.4
state = rk3_stepper(state, r.to(code_units.code_length).value, dx, alpha, gamma, C_cfl, dt_max, t_end)

# =================================================================

# =========================== PLotting ============================

# mass of the star
M_star = 40 * u.M_sun

num_cells = 10

# wind features
StellarWindFinalVelocity = 2000 * u.km / u.s
StellarWindMassLossRate = 2.965e-3 / (1e6 * u.yr) * M_star

v_wind = StellarWindFinalVelocity
M_dot = StellarWindMassLossRate
rho_0 = 2 * m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

# get weaver solution
weaver = Weaver(v_wind, M_dot, rho_0, p_0)

current_time = t_final

# density
r_density_weaver, density_weaver = weaver.get_density_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
r_density_weaver = r_density_weaver.to(u.parsec)
density_weaver = (density_weaver / m_p).to(u.cm**-3)

# velocity
r_velocity_weaver, velocity_weaver = weaver.get_velocity_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
r_velocity_weaver = r_velocity_weaver.to(u.parsec)
velocity_weaver = velocity_weaver.to(u.km / u.s)

# pressure
r_pressure_weaver, pressure_weaver = weaver.get_pressure_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
r_pressure_weaver = r_pressure_weaver.to(u.parsec)
pressure_weaver = (pressure_weaver / c.k_B).to(u.cm**-3 * u.K)


# plot final state, density, velocity, pressure, internal energy
rho = state[0, 1:] * code_units.code_density
vel = state[1, 1:] / state[0, 1:] * code_units.code_velocity
E = state[2, 1:] * code_units.code_pressure
internal_energy = calculate_internal_energy_from_energy(E, rho, vel)
p = calculate_pressure_from_internal_energy(internal_energy, rho, gamma)
entropic_function = p / state[0, 1:]**gamma

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

r = r[1:]

axs[0].set_yscale("log")
axs[0].plot(r.to(u.parsec), (rho / m_p).to(u.cm**-3), label="radial 1d simulation")
axs[0].plot(r_density_weaver, density_weaver, "--", label="Weaver solution")
axs[0].set_title("Density")
# xlim 0 to 0.3
# show legend
axs[0].legend()

axs[1].set_yscale("log")
axs[1].plot(r.to(u.parsec), (p / c.k_B).to(u.K / u.cm**3), label="radial 1d simulation")
axs[1].plot(r_pressure_weaver, pressure_weaver, "--", label="Weaver solution")
axs[1].set_title("Pressure")
# xlim 0 to 0.3
# show legend
axs[1].legend()

axs[2].set_yscale("log")
axs[2].plot(r.to(u.parsec), vel.to(u.km / u.s), label="radial 1d simulation")
axs[2].plot(r_velocity_weaver, velocity_weaver, "--", label="Weaver solution")
axs[2].set_title("Velocity")
# ylim 1 to 1e4 km/s
axs[2].set_ylim(1, 1e4)
# show legend
axs[2].legend()

# #axs[1, 1].plot(r, internal_energy / rho)
# # use log scale
# axs[1, 1].set_yscale("log")
# axs[1, 1].plot(r.to(u.parsec), internal_energy)
# axs[1, 1].set_title("internal energy")
# # xlim 0 to 0.3

# tight layout
plt.tight_layout()

plt.savefig("unit_test.png")

# =========================================================
