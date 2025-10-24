from autocvd import autocvd
autocvd(num_gpus = 1)

from jf1uids.initial_condition_generation.construct_primitive_state import construct_primitive_state

from jf1uids._physics_modules._cooling._cooling import get_effective_molecular_weights, get_pressure_from_temperature, get_temperature_from_pressure
from jf1uids._physics_modules._cooling._cooling_tables import schure_cooling

from jf1uids._physics_modules._stellar_wind.stellar_wind_options import EI, MEI, MEO

import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from jf1uids import WindParams
from jf1uids import SimulationConfig
from jf1uids import SimulationParams
from jf1uids.option_classes import WindConfig
from jf1uids._physics_modules._cooling.cooling_options import EXPLICIT_COOLING, IMPLICIT_COOLING, NEURAL_NET_COOLING, PIECEWISE_POWER_LAW, SIMPLE_POWER_LAW, CoolingConfig, CoolingCurveConfig, CoolingNetConfig, CoolingNetParams, CoolingParams, PiecewisePowerLawParams, SimplePowerLawParams

from jf1uids import get_helper_data
from jf1uids._fluid_equations._equations import conserved_state_from_primitive, primitive_state_from_conserved
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import BACKWARDS, FORWARDS, finalize_config

import pickle

from jf1uids import time_integration

# jf1uids constants
from jf1uids.option_classes.simulation_config import OPEN_BOUNDARY, REFLECTIVE_BOUNDARY, SPHERICAL

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from jf1uids._physics_modules._stellar_wind.weaver import Weaver

import equinox as eqx
import jax
import optax
from matplotlib.gridspec import GridSpec

# options
plot_problem_setting = True
run_simulation = True
high_res_s = [2000]
color_s = ["blue", "orange", "green"]
domain_size = 1.0
r_inj = 0.01 * domain_size

cooling_curve_type = PIECEWISE_POWER_LAW

def get_num_injection_cells(r_inj, num_cells):
    ngc = 0
    grid_spacing = domain_size / num_cells
    r = jnp.linspace(grid_spacing / 2 - ngc * grid_spacing, domain_size + grid_spacing / 2 + ngc * grid_spacing, num_cells + 2 * ngc, endpoint = False)
    outer_cell_boundaries = r + grid_spacing / 2
    r_inj = r_inj * domain_size
    return int(jnp.argmin(jnp.abs(outer_cell_boundaries - r_inj)))

# code units
code_length = 20 * u.pc
code_mass = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

t_final = 0.5 * 1e6 * u.yr
t_end = t_final.to(code_units.code_time).value
dt_max = 0.1 * t_end
M_star = 5000 * u.M_sun
wind_final_velocity = 3230 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 * M_star / (1e6 * u.yr)

print("wind_mass_loss_rate:", wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value)
print("wind_final_velocity:", wind_final_velocity.to(code_units.code_velocity).value)

# general cooling params
hydrogen_mass_fraction = 0.76
metal_mass_fraction = 0.02
# without a floor temperature, the simulations crash
floor_temperature = (1e2 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value

mu, mu_e, mu_H = get_effective_molecular_weights(
    hydrogen_mass_fraction,
    metal_mass_fraction
)

print("mu:", mu)
print("mu_e:", mu_e)
print("mu_H:", mu_H)

# homogeneous initial state
rho_0 = 86.25 * m_p / u.cm**3 * mu_H
p_0 = (6000 * u.K * c.k_B / u.cm**3).to(code_units.code_pressure)


schure_cooling_params = schure_cooling(code_units)

class CoolingNet(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        hidden_features: int,
        *,
        key
    ):
        self.mlp = eqx.nn.MLP(
            in_size=1,       # input: log10_T
            out_size=1,      # output: log10_Lambda
            width_size=hidden_features,
            depth=3,
            key=key,
        )
        
    def __call__(self, x):
        return self.mlp(x)
        

# initialize cooling corrector network
key = jax.random.PRNGKey(84)
cooling_corrector_network = CoolingNet(
    hidden_features = 256,
    key = key
)
_, cooling_corrector_static = eqx.partition(cooling_corrector_network, eqx.is_array)

net_config = CoolingNetConfig(
    network_static = cooling_corrector_static
)

with open("models/cooling_corrector10000_2000.pkl", "rb") as f:
    cooling_corrector_params = pickle.load(f)

net_curve_params = CoolingNetParams(
    network_params = cooling_corrector_params
)

cooling_curve_config = CoolingCurveConfig(
    cooling_curve_type = cooling_curve_type,
    cooling_net_config = CoolingNetConfig(
        network_static = cooling_corrector_static
    )
)

if cooling_curve_type == NEURAL_NET_COOLING:
    cooling_curve_params = net_curve_params
elif cooling_curve_type == PIECEWISE_POWER_LAW:
    cooling_curve_params = schure_cooling_params
else:
    raise ValueError("Invalid cooling curve type")

def setup_simulation(num_cells, cooling_curve_config, cooling_curve_params, return_snapshots, num_snapshots, cooling = True, num_injection_cells = 20, use_specific_snapshot_timepoints = True):
    print("👷 Setting up simulation...")

    # simulation settings
    gamma = 5/3

    # spatial domain
    geometry = SPHERICAL
    box_size = domain_size

    # activate stellar wind
    stellar_wind = True

    # time stepping
    C_CFL = 0.8

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging = False,
        progress_bar = True,
        geometry = geometry,
        box_size = box_size, 
        num_cells = num_cells,
        wind_config = WindConfig(
            stellar_wind = stellar_wind,
            num_injection_cells = num_injection_cells,
            trace_wind_density = False,
            wind_injection_scheme = EI
        ),
        cooling_config = CoolingConfig(
            cooling = cooling,
            cooling_method = EXPLICIT_COOLING,
            cooling_curve_config = cooling_curve_config
        ),
        return_snapshots = return_snapshots,
        num_snapshots = num_snapshots,
        use_specific_snapshot_timepoints = use_specific_snapshot_timepoints,
        differentiation_mode = FORWARDS,
        num_checkpoints = 100
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # wind parameters
    wind_params = WindParams(
        wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
        wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value
    )

    # simulation params
    params = SimulationParams(
        C_cfl = C_CFL,
        dt_max = dt_max,
        gamma = gamma,
        t_end = t_end,
        wind_params = wind_params,
        cooling_params = CoolingParams(
            hydrogen_mass_fraction = hydrogen_mass_fraction,
            metal_mass_fraction = metal_mass_fraction,
            floor_temperature = floor_temperature,
            cooling_curve_params = cooling_curve_params
        ),
        snapshot_timepoints = jnp.linspace(0, t_end, num_snapshots + 2)[2:]
    )

    rho_init = jnp.ones(num_cells) * rho_0.to(code_units.code_density).value
    u_init = jnp.zeros(num_cells)
    p_init = jnp.ones(num_cells) * p_0.to(code_units.code_pressure).value

    # get initial state
    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho_init,
        velocity_x = u_init,
        gas_pressure = p_init
    )

    config = finalize_config(config, initial_state.shape)

    return initial_state, config, params, helper_data, registered_variables


# compare with weaver solution
def plot_profiles(axs, final_state, registered_variables, helper_data, code_units, label = "jf1uids", left_gray = False, start_index = 0, color = "blue"):
    print("👷 generating plots")

    rho = final_state[registered_variables.density_index]
    vel = final_state[registered_variables.velocity_index]
    p = final_state[registered_variables.pressure_index]

    rho = rho * code_units.code_density
    vel = vel * code_units.code_velocity
    p = p * code_units.code_pressure

    r = helper_data.geometric_centers * code_units.code_length

    # temperature

    T = get_temperature_from_pressure(rho, p * c.m_p / c.k_B, hydrogen_mass_fraction, metal_mass_fraction)
    T = T.to(u.K)

    axs[0].set_yscale("log")
    axs[0].plot(r.to(u.pc), (rho).to(m_p * u.cm**-3), label=label, color = color)
    axs[0].set_title("density")
    axs[0].set_ylabel(r"$\rho$ in m$_p$ cm$^{-3}$")
    # axs[0].set_ylim(1e-27, 1e-21)
    # axs[0].set_xlim(0, 1e19)
    axs[0].legend(loc="lower right")
    axs[0].set_xlabel("r in pc")

    axs[1].set_yscale("log")
    axs[1].plot(r.to(u.pc), (p / c.k_B).to(u.K / u.cm**3), label=label, color = color)
    axs[1].set_title("pressure")
    axs[1].set_ylabel(r"$p$/k$_b$ in K cm$^{-3}$")
    # axs[1].set_xlim(0, 1e19)
    axs[1].legend(loc="upper left")
    axs[1].set_xlabel("r in pc")

    axs[2].set_yscale("log")
    axs[2].plot(r.to(u.pc), vel.to(u.km / u.s), label=label, color = color)
    axs[2].set_title("velocity")
    # axs[2].set_ylim(1, 1e4)
    # axs[2].set_xlim(0, 1e19)
    axs[2].set_ylabel("v in km/s")
    axs[2].legend(loc="upper right")
    axs[2].set_xlabel("r in pc")

    axs[3].set_yscale("log")
    axs[3].plot(r.to(u.pc), T.to(u.K), label=label, color = color)
    axs[3].set_title("temperature")
    # axs[3].set_ylim(10, 1e9)
    # axs[3].set_xlim(0, 1e19)
    axs[3].set_ylabel("T in K")
    axs[3].legend(loc="upper right")
    axs[3].set_xlabel("r in pc")

    if left_gray:
        # add gray background to r < r[start_index]
        axs[0].axvspan(0, r[start_index].to(u.pc).value, color="gray", alpha=0.3)
        axs[1].axvspan(0, r[start_index].to(u.pc).value, color="gray", alpha=0.3)
        axs[2].axvspan(0, r[start_index].to(u.pc).value, color="gray", alpha=0.3)
        axs[3].axvspan(0, r[start_index].to(u.pc).value, color="gray", alpha=0.3)

reference_state_collection = []

for high_res in high_res_s:
    # get reference simulation
    initial_state, config, params, helper_data, registered_variables = setup_simulation(
        num_cells = high_res,
        cooling_curve_config = cooling_curve_config,
        cooling_curve_params = cooling_curve_params,
        return_snapshots = True,
        num_snapshots = 5,
        num_injection_cells = get_num_injection_cells(r_inj, high_res)
    )

    if run_simulation:
        result = time_integration(initial_state, config, params, helper_data, registered_variables)
        reference_states = result.states
        # save reference states as numpy array
        jnp.save(f"data/reference_states{high_res}.npy", jnp.array(reference_states))
        
        # also run without cooling for comparison
        config_no_cooling = config._replace(
            cooling_config = CoolingConfig(
                cooling = False,
            )
        )
        result_no_cooling = time_integration(initial_state, config_no_cooling, params, helper_data, registered_variables)
        reference_states_no_cooling = result_no_cooling.states
        jnp.save(f"data/reference_states_no_cooling{high_res}.npy", jnp.array(reference_states_no_cooling))
    else:
        reference_states = jnp.load(f"data/reference_states{high_res}.npy", allow_pickle=True)

    reference_state_collection.append(reference_states)

# problem setting
if plot_problem_setting:

    # one ax with cooling, one without
    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    for high_res, color in zip(high_res_s, color_s):

        num_cells = high_res
        
        # setup simulation without cooling
        initial_state, config, params, helper_data, registered_variables = setup_simulation(
            num_cells = num_cells,
            cooling_curve_config = cooling_curve_config,
            cooling_curve_params = cooling_curve_params,
            return_snapshots = False,
            num_snapshots = 0,
            cooling = False,
        )
        final_state = jnp.load(f"data/reference_states_no_cooling{high_res}.npy", allow_pickle=True)[-1]

        plot_profiles(axs[0, :], final_state, registered_variables, helper_data, code_units, label = f"jf1uids, {high_res} cells", color = color)

        # setup simulation with cooling
        initial_state, config, params, helper_data, registered_variables = setup_simulation(
            num_cells = num_cells,
            cooling_curve_config = cooling_curve_config,
            cooling_curve_params = cooling_curve_params,
            return_snapshots = False,
            num_snapshots = 0,
            cooling = True,
        )

        final_state = jnp.load(f"data/reference_states{high_res}.npy", allow_pickle=True)[-1]

        plot_profiles(axs[1, :], final_state, registered_variables, helper_data, code_units, label = f"jf1uids, {high_res} cells", color = color)

    plt.tight_layout()
    plt.savefig("figures/problem_setting.svg")