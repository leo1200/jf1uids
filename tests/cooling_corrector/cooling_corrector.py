# TODO: fix units

from autocvd import autocvd

autocvd(num_gpus = 1)

from jf1uids._physics_modules._cooling._cooling import get_pressure_from_temperature, get_temperature_from_pressure
from jf1uids._physics_modules._cooling._cooling_tables import schure_cooling

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
from jf1uids._physics_modules._cooling.cooling_options import NEURAL_NET_COOLING, PIECEWISE_POWER_LAW, SIMPLE_POWER_LAW, CoolingConfig, CoolingCurveConfig, CoolingNetConfig, CoolingNetParams, CoolingParams, PiecewisePowerLawParams, SimplePowerLawParams

from jf1uids import get_helper_data
from jf1uids.fluid_equations.fluid import conserved_state_from_primitive, construct_primitive_state, primitive_state_from_conserved
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import BACKWARDS, finalize_config

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

# code units
code_length = 10e18 * u.cm
code_mass = 1e-3 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# low resolution simulation
low_res = 625
print("LOW RES = ", low_res)

# beginning index of loss
beginning_index = low_res // 4

# general cooling params
hydrogen_mass_fraction = 0.76
metal_mass_fraction = 0.02
# without a floor temperature, the simulations crash
floor_temperature = (1e2 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value


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
key = jax.random.PRNGKey(42)
cooling_corrector_network = CoolingNet(
    hidden_features = 256,
    key = key
)
cooling_corrector_params, cooling_corrector_static = eqx.partition(cooling_corrector_network, eqx.is_array)

# pre-train the network to log10_Lambda_table(log10_T_table)
# from the schure cooling table
schure_cooling_params = schure_cooling(code_units)
log10_T_table = schure_cooling_params.log10_T_table
log10_Lambda_table = schure_cooling_params.log10_Lambda_table
log10_T_table = log10_T_table[:, None]
log10_Lambda_table = log10_Lambda_table[:, None]

# print(log10_Lambda_table)
# print(log10_T_table)

# simple training loop
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(cooling_corrector_params)

@eqx.filter_jit
def loss_fn_pre_train(network_params_arrays):
    """Calculates the difference between the final state and the target."""
    network = eqx.combine(network_params_arrays, cooling_corrector_static)
    log10_Lambda_pred = jax.vmap(network)(log10_T_table)
    loss_value = jnp.mean((log10_Lambda_pred - log10_Lambda_table) ** 2)
    return loss_value

@eqx.filter_jit
def train_step_pre_train(network_params_arrays, opt_state):
    """Performs one step of gradient descent."""
    loss_value, grads = eqx.filter_value_and_grad(loss_fn_pre_train)(network_params_arrays)
    updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
    network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
    return network_params_arrays, opt_state, loss_value

num_epochs = 10000
for epoch in range(num_epochs):
    cooling_corrector_params, opt_state, loss_value = train_step_pre_train(cooling_corrector_params, opt_state)
    if epoch % 500 == 0 or epoch == num_epochs - 1:
        print(f"Pre-training Epoch {epoch+1}/{num_epochs}, Loss: {loss_value:.6f}")

cooling_corrector_network = eqx.combine(cooling_corrector_params, cooling_corrector_static)

cooling_curve_paramsC = CoolingNetParams(
    network_params = cooling_corrector_params
)

# plot the cooling curve
Lambda_net = jax.vmap(cooling_corrector_network)(log10_T_table)

plt.figure(figsize=(8, 6))
plt.plot(log10_T_table, log10_Lambda_table, label="Schure et al. 2009", color="blue")
plt.plot(log10_T_table, Lambda_net, label="Cooling Corrector NN", linestyle="--", color="orange")
plt.xlabel("Temperature")
plt.ylabel("Cooling Function Λ(T)")
plt.title("Cooling Function Comparison")
plt.legend()
plt.savefig("figures/cooling_function_comparison.png")


def setup_simulation(num_cells, cooling_curve_type, cooling_curve_params, return_snapshots, num_snapshots, cooling = True, num_injection_cells = 10, use_specific_snapshot_timepoints = True, t_final = 1.25e12 * u.s):
    print("👷 Setting up simulation...")

    # simulation settings
    gamma = 5/3

    # spatial domain
    geometry = SPHERICAL
    box_size = 1.0

    # activate stellar wind
    stellar_wind = True

    # time stepping
    C_CFL = 0.8
    t_end = t_final.to(code_units.code_time).value
    dt_max = 0.1 * t_end

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging = False,
        progress_bar = False,
        geometry = geometry,
        box_size = box_size, 
        num_cells = num_cells,
        wind_config = WindConfig(
            stellar_wind = stellar_wind,
            num_injection_cells = num_injection_cells,
            trace_wind_density = False,
        ),
        cooling_config = CoolingConfig(
            cooling = cooling,
            cooling_curve_config = CoolingCurveConfig(
                cooling_curve_type = cooling_curve_type,
                cooling_net_config = CoolingNetConfig(
                    network_static = cooling_corrector_static
                )
            )
        ),
        return_snapshots = return_snapshots,
        num_snapshots = num_snapshots,
        use_specific_snapshot_timepoints = use_specific_snapshot_timepoints,
        differentiation_mode = BACKWARDS,
        num_checkpoints = 100
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # wind parameters
    wind_final_velocity = 1500 * u.km / u.s
    wind_mass_loss_rate = 1e-6 * u.M_sun / u.yr
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

    # homogeneous initial state
    rho_0 = 10 ** (-22.5) * u.g / u.cm**3
    T_0 = 100 * u.K * c.k_B / c.m_p
    p_0 = get_pressure_from_temperature(rho_0, T_0, hydrogen_mass_fraction, metal_mass_fraction)
    print(p_0)

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

def plot_density(ax, state, registered_variables, helper_data, code_units, label):
    rho = state[registered_variables.density_index]
    rho = rho * code_units.code_density
    r = helper_data.geometric_centers * code_units.code_length
    ax.plot(r.to(u.cm), (rho).to(u.g * u.cm**-3), label=label)
    ax.set_yscale("log")
    ax.set_title("density")
    ax.set_ylabel(r"$\rho$ in g cm$^{-3}$")
    ax.set_ylim(1e-27, 1e-21)
    ax.set_xlim(0, 1e19)
    ax.set_xlabel("r in cm")


# compare with weaver solution
def plot_profiles(axs, final_state, registered_variables, helper_data, code_units, label = "jf1uids", left_gray = False, start_index = 0):
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
    axs[0].plot(r.to(u.cm), (rho).to(u.g * u.cm**-3), label=label)
    axs[0].set_title("density")
    axs[0].set_ylabel(r"$\rho$ in m$_p$ cm$^{-3}$")
    axs[0].set_ylim(1e-27, 1e-21)
    axs[0].set_xlim(0, 1e19)
    axs[0].legend(loc="lower right")
    axs[0].set_xlabel("r in cm")

    axs[1].set_yscale("log")
    axs[1].plot(r.to(u.cm), (p / c.k_B).to(u.K / u.cm**3), label=label)
    axs[1].set_title("pressure")
    axs[1].set_ylabel(r"$p$/k$_b$ in K cm$^{-3}$")
    axs[1].set_xlim(0, 1e19)
    axs[1].legend(loc="upper left")
    axs[1].set_xlabel("r in cm")

    axs[2].set_yscale("log")
    axs[2].plot(r.to(u.cm), vel.to(u.km / u.s), label=label)
    axs[2].set_title("velocity")
    axs[2].set_ylim(1, 1e4)
    axs[2].set_xlim(0, 1e19)
    axs[2].set_ylabel("v in km/s")
    axs[2].legend(loc="upper right")
    axs[2].set_xlabel("r in cm")

    axs[3].set_yscale("log")
    axs[3].plot(r.to(u.cm), T.to(u.K), label=label)
    axs[3].set_title("temperature")
    axs[3].set_ylim(10, 1e9)
    axs[3].set_xlim(0, 1e19)
    axs[3].set_ylabel("T in K")
    axs[3].legend(loc="upper right")
    axs[3].set_xlabel("r in cm")

    if left_gray:
        # add gray background to r < r[start_index]
        axs[0].axvspan(0, r[start_index].to(u.cm).value, color="gray", alpha=0.3)
        axs[1].axvspan(0, r[start_index].to(u.cm).value, color="gray", alpha=0.3)
        axs[2].axvspan(0, r[start_index].to(u.cm).value, color="gray", alpha=0.3)
        axs[3].axvspan(0, r[start_index].to(u.cm).value, color="gray", alpha=0.3)

# problem setting
plot_problem_setting = False
if plot_problem_setting:
    cell_nums = [500, 1000, 2000, 10000]
    # one ax with cooling, one without
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for num_cells in cell_nums:
        # setup simulation without cooling
        initial_state, config, params, helper_data, registered_variables = setup_simulation(
            num_cells = num_cells,
            cooling_curve_type = PIECEWISE_POWER_LAW,
            cooling_curve_params = schure_cooling_params,
            return_snapshots = False,
            num_snapshots = 0,
            cooling = False,
            num_injection_cells = 10 # 5 * int(num_cells / 500)
        )
        # run simulation without cooling
        final_state = time_integration(initial_state, config, params, helper_data, registered_variables)
        plot_density(axs[0], final_state, registered_variables, helper_data, code_units, label = f"{num_cells} cells")

        # setup simulation with cooling
        initial_state, config, params, helper_data, registered_variables = setup_simulation(
            num_cells = num_cells,
            cooling_curve_type = PIECEWISE_POWER_LAW,
            cooling_curve_params = schure_cooling_params,
            return_snapshots = False,
            num_snapshots = 0,
            cooling = True,
            num_injection_cells = 10 # 5 * int(num_cells / 500)
        )
        # run simulation with cooling
        final_state = time_integration(initial_state, config, params, helper_data, registered_variables)
        plot_density(axs[1], final_state, registered_variables, helper_data, code_units, label = f"{num_cells} cells")
    axs[0].set_title("without cooling")
    axs[1].set_title("with cooling")
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.savefig("figures/problem_setting.svg")

# get reference simulation
high_res = 10000
run_simulation = True
initial_state, config, params, helper_data, registered_variables = setup_simulation(
    num_cells = high_res,
    cooling_curve_type = PIECEWISE_POWER_LAW,
    cooling_curve_params = schure_cooling_params,
    return_snapshots = True,
    num_snapshots = 5
)

if run_simulation:
    result = time_integration(initial_state, config, params, helper_data, registered_variables)
    reference_states = result.states

    # save reference states as numpy array
    jnp.save("data/reference_states.npy", jnp.array(reference_states))
else:
    reference_states = jnp.load("data/reference_states.npy", allow_pickle=True)

print(reference_states.shape)

final_state = reference_states[-1]
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
plot_profiles(axs, final_state, registered_variables, helper_data, code_units)
plt.tight_layout()
plt.savefig("figures/reference_output.svg")

# sth wrong
# # downsample the reference states to low_res, states have shape: num_vars x num_cells
# reference_states_downsampled = jnp.zeros((reference_states.shape[0], initial_state.shape[0], low_res))
# for i in range(reference_states.shape[0]):

#     cell_volumes = helper_data.cell_volumes
#     low_res_cell_volumes = jnp.sum(jnp.reshape(cell_volumes, (-1, low_res)), axis=0)

#     conservative_state = conserved_state_from_primitive(
#         primitive_state = reference_states[i],
#         gamma = params.gamma,
#         config = config,
#         registered_variables = registered_variables
#     )
#     absolute_quantities = conservative_state * cell_volumes[None, :]
#     low_res_absolute_quantities = jnp.sum(jnp.reshape(absolute_quantities, (reference_states[i].shape[0], low_res, -1)), axis=-1)
#     low_res_conservative_state = low_res_absolute_quantities / low_res_cell_volumes[None, :]

#     low_res_primitive_state = primitive_state_from_conserved(
#         conserved_state = low_res_conservative_state,
#         gamma = params.gamma,
#         config = config,
#         registered_variables = registered_variables
#     )

#     reference_states_downsampled = reference_states_downsampled.at[i].set(low_res_primitive_state)

reference_states_downsampled = jnp.mean(jnp.reshape(reference_states, (reference_states.shape[0], reference_states.shape[1], low_res, -1)), axis = -1)

initial_state, config, params, helper_data_low_res, registered_variables = setup_simulation(
    num_cells = low_res,
    cooling_curve_type = NEURAL_NET_COOLING,
    cooling_curve_params = cooling_curve_paramsC,
    return_snapshots = True,
    num_snapshots = 5
)

learning_reference_timepoints = (params.snapshot_timepoints * code_units.code_time).to(u.yr).value

# plot downsampled result
final_state_downsampled = reference_states_downsampled[-1]
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
plot_profiles(axs, final_state_downsampled, registered_variables, helper_data_low_res, code_units)
plt.tight_layout()
plt.savefig("figures/downsampled_reference_output.svg")

# plot low res simulation result
result = time_integration(initial_state, config, params, helper_data_low_res, registered_variables)
low_res_states = result.states
final_state = low_res_states[-1]
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
plot_profiles(axs, final_state, registered_variables, helper_data_low_res, code_units)
plt.tight_layout()
plt.savefig("figures/low_res_output.svg")

# loss function
@eqx.filter_jit
def loss_fn_train(network_params):
    """Calculates the difference between the final state and the target."""
    params_new = params._replace(
        cooling_params = params.cooling_params._replace(
            cooling_curve_params = CoolingNetParams(
                network_params = network_params
            )
        )
    )
    result = time_integration(initial_state, config, params_new, helper_data_low_res, registered_variables)
    return jnp.mean(((result.states[:, :, beginning_index:] - reference_states_downsampled[:, :, beginning_index:]) / jnp.max(reference_states_downsampled[:, :, beginning_index:], axis = (0, 2))[None, :, None]) ** 2)


@eqx.filter_jit
def train_step_train(network_params_arrays, opt_state):
    """Performs one step of gradient descent."""
    loss_value, grads = eqx.filter_value_and_grad(loss_fn_train)(network_params_arrays)
    updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
    network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
    return network_params_arrays, opt_state, loss_value


def plot_loss(network_params, params, frame, post_string = ""):
    params = params._replace(
        cooling_params = params.cooling_params._replace(
            cooling_curve_params = CoolingNetParams(
                network_params = network_params
            )
        )
    )
    result = time_integration(initial_state, config, params, helper_data_low_res, registered_variables)
    loss = ((result.states - reference_states_downsampled) / jnp.max(reference_states_downsampled, axis = (0, 2))[None, :, None]) ** 2
    loss_f = loss[frame]
    # plot density, velocity and pressure loss
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(loss_f[0, beginning_index:])
    axs[1].plot(loss_f[1, beginning_index:])
    axs[2].plot(loss_f[2, beginning_index:])
    axs[0].set_title("Density Loss")
    axs[1].set_title("Velocity Loss")
    axs[2].set_title("Pressure Loss")
    plt.tight_layout()
    plt.savefig(f"figures/loss_frame_{frame}{post_string}.png")

# training loop
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(cooling_corrector_params)
num_epochs = 150

plot_loss(cooling_corrector_params, params, frame = -1)
print(loss_fn_train(cooling_corrector_params))

best_loss = jnp.inf
best_params = cooling_corrector_params
train_model = True

if train_model:
    for epoch in range(num_epochs):
        cooling_corrector_params, opt_state, loss_value = train_step_train(cooling_corrector_params, opt_state)

        if loss_value < best_loss:
            best_loss = loss_value
            best_params = cooling_corrector_params

        if epoch % 1 == 0 or epoch == num_epochs - 1:
            print(f"Training Epoch {epoch+1}/{num_epochs}, Loss: {loss_value:.6f}")

    # use best params
    cooling_corrector_params = best_params

    # save the trained network parameters with pickle
    with open("models/cooling_corrector" + str(low_res) + ".pkl", "wb") as f:
        pickle.dump(cooling_corrector_params, f)
else:
    # load the trained network parameters with pickle
    # import pickle
    with open("models/cooling_corrector" + str(low_res) + ".pkl", "rb") as f:
        cooling_corrector_params = pickle.load(f)


plot_loss(cooling_corrector_params, params, frame = -1, post_string = "post_training")

# plot low res simulation result after training
params = params._replace(
    cooling_params = params.cooling_params._replace(
        cooling_curve_params = CoolingNetParams(
            network_params = cooling_corrector_params
        )
    )
)
result = time_integration(initial_state, config, params, helper_data_low_res, registered_variables)
low_res_states_corrected = result.states
final_state = low_res_states_corrected[-1]
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
plot_profiles(axs, final_state, registered_variables, helper_data_low_res, code_units)
plt.tight_layout()
plt.savefig("figures/low_res_output_corrected.svg")

# compare cooling curves
cooling_corrector_network = eqx.combine(cooling_corrector_params, cooling_corrector_static)
Lambda_net_after_training = jax.vmap(cooling_corrector_network)(log10_T_table)
plt.figure(figsize=(8, 6))
plt.plot(log10_T_table, log10_Lambda_table, label="Schure et al. 2009", color="blue")
plt.plot(log10_T_table, Lambda_net, label="Cooling Corrector NN (pre-training)", linestyle="--", color="orange")
plt.plot(log10_T_table, Lambda_net_after_training, label="Cooling Corrector NN (after training)", linestyle=":", color="green")
plt.xlabel("temperature")
plt.ylabel("cooling function Λ(T)")
plt.title("Cooling Function Comparison")
plt.legend()
plt.savefig("figures/final_cooling_functions.svg")

# compare the reference to the uncorrected and corrected low res simulation

# run simulations longer and a t higher time resolution

# reference
initial_state, config, params, helper_data_low_res, registered_variables = setup_simulation(
    num_cells = high_res,
    cooling_curve_type = PIECEWISE_POWER_LAW,
    cooling_curve_params = schure_cooling_params,
    return_snapshots = True,
    num_snapshots = 200,
    t_final = 1.3e12 * u.s
)
err_timepoints = (params.snapshot_timepoints * code_units.code_time).to(u.yr).value
result = time_integration(initial_state, config, params, helper_data_low_res, registered_variables)
reference_states = result.states
reference_states_downwampled = jnp.mean(jnp.reshape(reference_states, (reference_states.shape[0], reference_states.shape[1], low_res, -1)), axis = -1)

# uncorrected low res
initial_state, config, params, helper_data_low_res, registered_variables = setup_simulation(
    num_cells = low_res,
    cooling_curve_type = PIECEWISE_POWER_LAW,
    cooling_curve_params = schure_cooling_params,
    return_snapshots = True,
    num_snapshots = 200,
    t_final = 1.3e12 * u.s
)
result = time_integration(initial_state, config, params, helper_data_low_res, registered_variables)
low_res_states = result.states
error_uncorrected = jnp.mean(((low_res_states[:, :, beginning_index:] - reference_states_downwampled[:, :, beginning_index:]) / jnp.max(reference_states_downwampled[:, :, beginning_index:], axis = (0, 2))[None, :, None]) ** 2, axis = (1, 2))

# corrected low res
initial_state, config, params, helper_data_low_res, registered_variables = setup_simulation(
    num_cells = low_res,
    cooling_curve_type = NEURAL_NET_COOLING,
    cooling_curve_params = CoolingNetParams(
        network_params = cooling_corrector_params
    ),
    return_snapshots = True,
    num_snapshots = 200,
    t_final = 1.3e12 * u.s
)
result = time_integration(initial_state, config, params, helper_data_low_res, registered_variables)
low_res_states_corrected = result.states
error_corrected = jnp.mean(((low_res_states_corrected[:, :, beginning_index:] - reference_states_downwampled[:, :, beginning_index:]) / jnp.max(reference_states_downwampled[:, :, beginning_index:], axis = (0, 2))[None, :, None]) ** 2, axis = (1, 2))

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 4, height_ratios=[2, 1])

# Top row: profiles
axs_profiles = [fig.add_subplot(gs[0, i]) for i in range(4)]
plot_profiles(axs_profiles, reference_states_downwampled[-1], registered_variables, helper_data_low_res, code_units, label="reference (10k cells)")
plot_profiles(axs_profiles, low_res_states[-1], registered_variables, helper_data_low_res, code_units, label="uncorrected (500 cells)")
plot_profiles(axs_profiles, low_res_states_corrected[-1], registered_variables, helper_data_low_res, code_units, label="low res corrected (500 cells)", start_index = beginning_index, left_gray = True)

# Bottom row: error plot
ax_error = fig.add_subplot(gs[1, :])
ax_error.plot(err_timepoints, error_uncorrected, label="uncorrected", color="red")
ax_error.plot(err_timepoints, error_corrected, label="corrected", color="green")
for i, t in enumerate(learning_reference_timepoints):
    if i == 0:
        ax_error.axvline(t, color="gray", linestyle="--", alpha=0.5, label="reference timepoints")
    else:
        ax_error.axvline(t, color="gray", linestyle="--", alpha=0.5)
ax_error.set_xlabel("time in years")
ax_error.set_ylabel("mean squared error")
ax_error.set_title("error over time")
ax_error.legend()

plt.tight_layout()
plt.savefig("figures/final_comparison" + str(low_res) + ".svg")