# # ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1, interval=10)
# # =======================

from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector import CorrectorCNN
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDParams,
    CNNMHDconfig,
)

# numerics
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


from timeit import default_timer as timer
from tqdm import tqdm
import os

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# equinox
import equinox as eqx

# optax for optimization
import optax

# fluids
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# jf1uids constants
from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    FORWARDS,
    HLL,
    HLLC,
    MINMOD,
    OSHER,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

from corrector_src.data.blast_creation import initial_blast_state
from corrector_src.utils.downaverage import downaverage_state


high_resolution = 512
low_resolution = 64

(
    initial_state_high_res,
    config_high_res,
    params_high_res,
    helper_data_high_res,
    registered_variables_high_res,
) = initial_blast_state(high_resolution)

target_state_high_res = time_integration(
    initial_state_high_res,
    config_high_res,
    params_high_res,
    helper_data_high_res,
    registered_variables_high_res,
)
target_state_low_res = downaverage_state(
    target_state_high_res, (low_resolution, low_resolution)
)

model = CorrectorCNN(
    in_channels=registered_variables_high_res.num_vars,
    hidden_channels=16,
    key=jax.random.PRNGKey(42),
)
neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

cnn_mhd_corrector_config = CNNMHDconfig(
    cnn_mhd_corrector=True, network_static=neural_net_static
)

cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

# initialize low res simulation
(
    initial_state_low_res,
    config_low_res,
    params_low_res,
    helper_data_low_res,
    registered_variables_low_res,
) = initial_blast_state(low_resolution)
config_low_res = config_low_res._replace(
    cnn_mhd_corrector_config=cnn_mhd_corrector_config
)
params_low_res = params_low_res._replace(
    cnn_mhd_corrector_params=cnn_mhd_corrector_params
)


@eqx.filter_jit
def loss_fn(network_params_arrays):
    """Calculates the difference between the final state and the target."""
    final_state_low_res = time_integration(
        initial_state_low_res,
        config_low_res,
        params_low_res._replace(
            cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                network_params=network_params_arrays
            )
        ),
        helper_data_low_res,
        registered_variables_low_res,
    )
    # Calculate the L2 loss between the final state and the target state
    loss = jnp.mean((final_state_low_res - target_state_low_res) ** 2)
    return loss


# # Set up the optimizer using optax
# learning_rate = 1e-3
# optimizer = optax.adam(learning_rate)
# opt_state = optimizer.init(neural_net_params)

# @eqx.filter_jit
# def train_step(network_params_arrays, opt_state):
#     """Performs one step of gradient descent."""
#     loss_value, grads = eqx.filter_value_and_grad(loss_fn)(network_params_arrays)
#     updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
#     network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
#     return network_params_arrays, opt_state, loss_value

# # ===================================================
# # ================== ↓ Training Loop ↓ ==============
# # ===================================================
# print("Starting training with optax...")
# num_steps = 5000
# losses = []

# # This variable will hold the trained parameters and be updated in the loop
# trained_params = neural_net_params

# # Timing
# start_time = timer()

# # The main training loop
# pbar = tqdm(range(num_steps))
# best_loss = float('inf')
# best_params = trained_params
# for step in pbar:
#     trained_params, opt_state, loss = train_step(trained_params, opt_state)
#     losses.append(loss)
#     if loss < best_loss:
#         best_loss = loss
#         best_params = trained_params
#     pbar.set_description(f"Step {step+1}/{num_steps} | Loss: {loss:.2e}")

# # After training, use the best parameters found
# trained_params = best_params

# end_time = timer()
# print(f"Training finished in {end_time - start_time:.2f} seconds.")
# # ===================================================
# # ================== ↑ Training Loop ↑ ==============
# # ===================================================

# pickle the trained parameters for later use
import pickle
# with open("trained_paramsX.pkl", "wb") as f:
#     pickle.dump(trained_params, f)

# load the trained parameters if needed
with open("trained_paramsX.pkl", "rb") as f:
    trained_params = pickle.load(f)

# # save the losses for later analysis with jnp.savez
# jnp.savez("lossesX.npz", losses = losses)

# plot the target state and the final state after training and the
# final state before training
final_state_low_res = time_integration(
    initial_state_low_res,
    config_low_res,
    params_low_res._replace(
        cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
            network_params=trained_params
        )
    ),
    helper_data_low_res,
    registered_variables_low_res,
)

(
    initial_state_low_res_uncorrected,
    config_low_res_uncorrected,
    params_low_res_uncorrected,
    helper_data_low_res_uncorrected,
    registered_variables_low_res_uncorrected,
) = get_blast_setup(low_resolution)
final_state_low_res_uncorrected = time_integration(
    initial_state_low_res_uncorrected,
    config_low_res_uncorrected,
    params_low_res_uncorrected,
    helper_data_low_res_uncorrected,
    registered_variables_low_res_uncorrected,
)
# load the losses from the file
losses_data = np.load("lossesX.npz")
losses = losses_data["losses"]


def get_errors_over_time(time):
    # first do the high resolution simulation
    (
        initial_state_high_res,
        config_high_res,
        params_high_res,
        helper_data_high_res,
        registered_variables_high_res,
    ) = get_blast_setup(high_resolution)
    final_state_high_res = time_integration(
        initial_state_high_res,
        config_high_res,
        params_high_res._replace(t_end=time),
        helper_data_high_res,
        registered_variables_high_res,
    )
    # downaverage the final state to low resolution
    final_state_high_res_downsampled = downaverage_state(
        final_state_high_res, (low_resolution, low_resolution)
    )

    # now do the uncorrected low resolution simulation
    (
        initial_state_low_res_uncorrected,
        config_low_res_uncorrected,
        params_low_res_uncorrected,
        helper_data_low_res_uncorrected,
        registered_variables_low_res_uncorrected,
    ) = get_blast_setup(low_resolution)
    final_state_low_res_uncorrected = time_integration(
        initial_state_low_res_uncorrected,
        config_low_res_uncorrected,
        params_low_res_uncorrected._replace(t_end=time),
        helper_data_low_res_uncorrected,
        registered_variables_low_res_uncorrected,
    )

    # now do the corrected low resolution simulation
    # the config is alredy correctly setup
    final_state_low_res_corrected = time_integration(
        initial_state_low_res,
        config_low_res,
        params_low_res._replace(
            cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                network_params=trained_params
            ),
            t_end=time,
        ),
        helper_data_low_res,
        registered_variables_low_res,
    )

    # calculate the L2 errors
    l2_error_corrected = jnp.mean(
        (final_state_low_res_corrected - final_state_high_res_downsampled) ** 2
    )
    l2_error_uncorrected = jnp.mean(
        (final_state_low_res_uncorrected - final_state_high_res_downsampled) ** 2
    )

    return l2_error_corrected, l2_error_uncorrected


# times = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.175, 0.2, 0.225, 0.25]
# l2_errors_corrected = []
# l2_errors_uncorrected = []
# for time in times:
#     l2_error_corrected, l2_error_uncorrected = get_errors_over_time(time)
#     l2_errors_corrected.append(l2_error_corrected)
#     l2_errors_uncorrected.append(l2_error_uncorrected)

# # save the errors to a file
# with open("l2_errors.pkl", "wb") as f:
#     pickle.dump({
#         "times": times,
#         "l2_errors_corrected": l2_errors_corrected,
#         "l2_errors_uncorrected": l2_errors_uncorrected
#     }, f)

# Load L2 error evolution over time
with open("l2_errors.pkl", "rb") as f:
    errors_data = pickle.load(f)
    times = errors_data["times"]
    l2_errors_corrected = errors_data["l2_errors_corrected"]
    l2_errors_uncorrected = errors_data["l2_errors_uncorrected"]

# Calculate initial L2 error
l2_error = jnp.mean((final_state_low_res_uncorrected - target_state_low_res) ** 2)
print(f"Initial L2 error (before training): {l2_error:.2e}")

# --- Create figure and layout ---
fig = plt.figure(figsize=(12, 12))
gs = GridSpec(3, 3, height_ratios=[1, 1, 1])

# First row: density images
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
titles = [
    "Target State (Density)",
    "Final State before Training (Density)",
    "Final State after Training (Density)",
]
states = [
    target_state_low_res,
    final_state_low_res_uncorrected,
    final_state_low_res,
]

# Shared color scale
vmin = min(jnp.min(s[registered_variables_low_res.density_index]) for s in states)
vmax = max(jnp.max(s[registered_variables_low_res.density_index]) for s in states)

for ax, state, title in zip(axs, states, titles):
    im = ax.imshow(
        state[registered_variables_low_res.density_index, ...],
        extent=(0, config_low_res.box_size, 0, config_low_res.box_size),
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Density")

# Second row: loss curve
ax_loss = fig.add_subplot(gs[1, :])
ax_loss.plot(losses, label="Training Loss")
ax_loss.axhline(
    y=l2_error, color="r", linestyle="--", label="Initial L2 Error (uncorrected)"
)
ax_loss.set_xlabel("Training Step")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Loss During Training")
ax_loss.legend()

# Third row: L2 error over time
ax_errors = fig.add_subplot(gs[2, :])
ax_errors.plot(
    times, l2_errors_corrected, label="Corrected Integration", color="tab:blue"
)
ax_errors.plot(
    times,
    l2_errors_uncorrected,
    label="Uncorrected Integration",
    color="tab:orange",
    linestyle="--",
)

# add an x-line at t = 0.2 (training reference time)
ax_errors.axvline(x=0.2, color="gray", linestyle=":", label="Training Reference Time")

ax_errors.set_xlabel("Time")
ax_errors.set_ylabel("L2 Error")
ax_errors.set_yscale("log")
ax_errors.set_title("Mean Squared Error Over Time")
ax_errors.legend()

plt.tight_layout()
plt.savefig("mhd_optimization.png", dpi=400)
