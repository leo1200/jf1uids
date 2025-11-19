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

configuration_names = ["fd_mhd", "fv_mhd"]

@partial(jax.jit, static_argnames=["target_shape"])
def downaverage_field(field: jnp.ndarray, target_shape: tuple[int, int, int]) -> jnp.ndarray:
    
    h_in, w_in, z_in = field.shape
    h_out, w_out, z_out = target_shape

    if h_in % h_out != 0 or w_in % w_out != 0 or z_in % z_out != 0:
        raise ValueError(
            f"Input shape {(h_in, w_in, z_in)} is not divisible by target shape {(h_out, w_out, z_out)}"
        )
    
    h_factor = h_in // h_out
    w_factor = w_in // w_out
    z_factor = z_in // z_out
    
    reshaped = field.reshape(h_out, h_factor, w_out, w_factor, z_out, z_factor)
    
    downaveraged = reshaped.mean(axis=(1, 3, 5))
    
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

        print(f"processing {configuration_name} at resolution {res}...")

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

axs[0].set_xlabel("resolution in number of cells per dimension")
axs[0].set_ylabel("total time in seconds")
axs[0].set_title("total time vs resolution")
axs[0].set_yscale("log")
axs[0].legend()

axs[1].set_xlabel("resolution in number of cells per dimension")
axs[1].set_ylabel("number of iterations")
axs[1].set_title("number of iterations vs resolution")
axs[1].legend()

axs[2].set_xlabel("resolution in number of cells per dimension")
axs[2].set_ylabel("time per iteration in seconds")
axs[2].set_yscale("log")
axs[2].set_title("time per iteration vs resolution")
axs[2].legend()

plt.tight_layout()
plt.savefig("scaling_comparison.svg")

# retrieve the reference final state for the highest resolution in how_mhd
reference_resolution = resolutions[-1]
reference_test_name = f"scaling_{reference_resolution}_{'multiGPU' if multi_gpu else 'singleGPU'}"
reference_output_folder = os.path.join("results", "fd_mhd", "data")
reference_output_file = os.path.join(reference_output_folder, reference_test_name + ".npz")
reference_data = jnp.load(reference_output_file)
reference_density = reference_data["final_state"][0]
reference_data = None  # free memory

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
        total_time = data["total_time"]

        final_state = data["final_state"]
        data = None  # free memory

        # downaverage the reference final state to the current resolution
        target_shape = (int(res), int(res), int(res))
        downsampled_reference_density = downaverage_field(reference_density, target_shape)

        # compute L2 error for the density field (assuming density is the first variable)
        density_final = final_state[0]
        final_state = None  # free memory

        l2_error = jnp.sqrt(jnp.mean((density_final - downsampled_reference_density) ** 2))
        l2_errors.append(l2_error)

        total_times.append(total_time)

    ax.plot(total_times, l2_errors, "o", label=configuration_name)
ax.set_xlabel("total time in seconds")
ax.set_ylabel("l2 error of density field")
ax.set_xscale("log")
ax.set_title("l2 error vs total time")
ax.legend()
plt.tight_layout()
plt.savefig("l2_error_comparison.svg")