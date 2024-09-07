import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax.numpy as jnp

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data

import matplotlib.pyplot as plt

# SETUP SIMULATION
input_manager = InputManager("sod.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables, \
forcing_parameters = initialization_manager.initialization()
# print(forcing_parameters)
sim_manager.simulate(simulation_buffers, time_control_variables)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "velocity", "pressure"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

def calculate_total_energy_from_primitive_vars(rho, u, p, gamma):
    """
    Calculate total energy from primitive variables.
    
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
        float
        Total energy per unit volume.
    """
    return p / (gamma - 1) + 0.5 * rho * u**2

def calculate_cell_volumes(cell_centers, dx):
    left_edges = cell_centers - dx / 2
    right_edges = cell_centers + dx / 2
    return 4/3 * jnp.pi * (right_edges**3 - left_edges**3)

def calculate_total(cell_volumes, field):
    return jnp.sum(cell_volumes * field)

# PLOT
plot_dict = {
    "density": data_dict["density"], 
    "velocityX": data_dict["velocity"][:,0],
    "pressure": data_dict["pressure"]
}

index = 15

density = plot_dict["density"][index, :, 0, 0]
velocity = plot_dict["velocityX"][index, :, 0, 0]
pressure = plot_dict["pressure"][index, :, 0, 0]


r = cell_centers[0]
dx = cell_sizes[0]

cell_volumes = calculate_cell_volumes(r, dx)
total_initial_mass = calculate_total(cell_volumes, plot_dict["density"][0, :, 0, 0])
initial_energy = calculate_total_energy_from_primitive_vars(plot_dict["density"][0, :, 0, 0], plot_dict["velocityX"][0, :, 0, 0], plot_dict["pressure"][0, :, 0, 0], 1.6666666)
total_initial_energy = calculate_total(cell_volumes, initial_energy)

mass_errors = []
energy_errors = []

for i in range(1, len(plot_dict["density"])):
    total_final_mass = calculate_total(cell_volumes, plot_dict["density"][i, :, 0, 0])
    final_energy = calculate_total_energy_from_primitive_vars(plot_dict["density"][i, :, 0, 0], plot_dict["velocityX"][i, :, 0, 0], plot_dict["pressure"][i, :, 0, 0], 1.6666666)
    total_final_energy = calculate_total(cell_volumes, final_energy)

    relative_error_total_mass = (total_final_mass - total_initial_mass) / total_initial_mass * 100
    relative_error_total_energy = (total_final_energy - total_initial_energy) / total_initial_energy * 100

    mass_errors.append(relative_error_total_mass)
    energy_errors.append(relative_error_total_energy)

mass_errors = jnp.array(mass_errors)
energy_errors = jnp.array(energy_errors)
times = times[1:]

# save the error to a numpy file
jnp.save("sod_errors.npy", jnp.array([times, mass_errors, energy_errors]))

print(f"Relative error in total mass in %: {mass_errors}")
print(f"Relative error in total energy in %: {energy_errors}")

# plot cell centers and densities
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(r, density)
axs[0].set_title("Density")
axs[0].set_xlabel("Cell centers")
axs[0].set_ylabel("Density")
axs[1].plot(r, velocity)
axs[1].set_title("Velocity")
axs[1].set_xlabel("Cell centers")
axs[1].set_ylabel("Velocity")
axs[2].plot(r, pressure)
axs[2].set_title("Pressure")
axs[2].set_xlabel("Cell centers")
axs[2].set_ylabel("Pressure")

plt.savefig("sod.png")