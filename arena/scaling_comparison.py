from functools import partial
import os

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

resolutions = [32, 64, 128, 256, 512]

multi_gpu = False

configuration_names = ["how_mhd", "fv_mhd"]

@partial(jax.jit, static_argnames=["target_shape"])
def downaverage_state(state: jnp.ndarray, target_shape: tuple[int, int, int]) -> jnp.ndarray:
    
    num_vars, h_in, w_in, z_in = state.shape
    h_out, w_out, z_out = target_shape

    if h_in % h_out != 0 or w_in % w_out != 0 or z_in % z_out != 0:
        raise ValueError(
            f"Input shape {(h_in, w_in, z_in)} is not divisible by target shape {(h_out, w_out, z_out)}"
        )
    
    h_factor = h_in // h_out
    w_factor = w_in // w_out
    z_factor = z_in // z_out
    
    reshaped = state.reshape(num_vars, h_out, h_factor, w_out, w_factor, z_out, z_factor)
    
    downaveraged = reshaped.mean(axis=(2, 4, 6))
    
    return downaveraged

# output_folder = os.path.join("results", configuration_name, "data")
# os.makedirs(output_folder, exist_ok=True)
# output_file = os.path.join(output_folder, test_name + ".npz")
# jnp.savez(
#     output_file,
#     final_state=final_state,
#     num_iterations=num_iterations,
#     total_time=total_time,
#     in_size_mb=in_size_mb,
#     temp_size_mb=temp_size_mb,
#     total_size_mb=total_size_mb,
# )


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for configuration_name in configuration_names:

    total_times = []
    num_iterations = []

    for res in resolutions:

        print(f"Processing {configuration_name} at resolution {res}...")

        test_name = f"scaling_{res}_{'multiGPU' if multi_gpu else 'singleGPU'}"

        output_folder = os.path.join("results", configuration_name, "data")
        output_file = os.path.join(output_folder, test_name + ".npz")
        data = np.load(output_file)

        total_times.append(data["total_time"])
        num_iterations.append(data["num_iterations"])

    total_times = np.array(total_times)
    num_iterations = np.array(num_iterations)
    time_per_iteration = total_times / num_iterations

    axs[0].plot(resolutions, total_times, marker='o', label=configuration_name)
    axs[1].plot(resolutions, num_iterations, marker='o', label=configuration_name)
    axs[2].plot(resolutions, time_per_iteration, marker='o', label=configuration_name)

axs[0].set_xlabel("Resolution (Number of Cells per Dimension)")
axs[0].set_ylabel("Total Time (s)")
axs[0].set_title("Total Time vs Resolution")
axs[0].set_yscale("log")
axs[0].legend()

axs[1].set_xlabel("Resolution (Number of Cells per Dimension)")
axs[1].set_ylabel("Number of Iterations")
axs[1].set_title("Number of Iterations vs Resolution")
axs[1].legend()

axs[2].set_xlabel("Resolution (Number of Cells per Dimension)")
axs[2].set_ylabel("Time per Iteration (s)")
axs[2].set_yscale("log")
axs[2].set_title("Time per Iteration vs Resolution")
axs[2].legend()

plt.tight_layout()
plt.savefig("scaling_comparison.svg")

# retrieve the reference final state for the highest resolution in how_mhd
reference_resolution = resolutions[-1]
reference_test_name = f"scaling_{reference_resolution}_{'multiGPU' if multi_gpu else 'singleGPU'}"
reference_output_folder = os.path.join("results", "how_mhd", "data")
reference_output_file = os.path.join(reference_output_folder, reference_test_name + ".npz")
reference_data = jnp.load(reference_output_file)
reference_final_state = reference_data["final_state"]

# plot L2 error of the density field compared to the reference solution
# plot L2 over total time
fig, ax = plt.subplots(figsize=(8, 6))

for configuration_name in configuration_names:
    
    l2_errors = []
    total_times = []

    for res in resolutions:

        test_name = f"scaling_{res}_{'multiGPU' if multi_gpu else 'singleGPU'}"

        print(f"Processing {configuration_name} at resolution {res}...")

        output_folder = os.path.join("results", configuration_name, "data")
        output_file = os.path.join(output_folder, test_name + ".npz")
        data = jnp.load(output_file)

        final_state = data["final_state"]

        # downaverage the reference final state to the current resolution
        target_shape = (int(res), int(res), int(res))
        downsampled_reference_state = downaverage_state(reference_final_state, target_shape)

        # compute L2 error for the density field (assuming density is the first variable)
        density_final = final_state[0]
        density_reference = downsampled_reference_state[0]

        l2_error = jnp.sqrt(jnp.mean((density_final - density_reference) ** 2))
        l2_errors.append(l2_error)

        total_times.append(data["total_time"])

    ax.plot(total_times, l2_errors, "o", label=configuration_name)
ax.set_xlabel("Total Time (s)")
ax.set_ylabel("L2 Error of Density Field")
ax.set_xscale("log")
ax.set_title("L2 Error vs Total Time")
ax.legend()
plt.tight_layout()
plt.savefig("l2_error_comparison.svg")