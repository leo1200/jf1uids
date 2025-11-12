# ==== GPU selection ====
from autocvd import autocvd

num_gpus = 1

autocvd(num_gpus=num_gpus)
# =======================

# numerics
from corrector_src.data.dataset import dataset
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib import animation

from jf1uids.fluid_equations.total_quantities import (
    calculate_internal_energy,
    calculate_kinetic_energy,
    calculate_total_energy,
    calculate_total_mass,
)
from omegaconf import OmegaConf

import numpy as np
from hydra import initialize, compose
from hydra.utils import instantiate
import equinox as eqx
import os
from corrector_src.model.fno_hd_force_corrector import TurbulenceSGSForceCorrectorFNO
from corrector_src.model._corrector_options import (
    CorrectorConfig,
    CorrectorParams,
)


def load_config():
    with initialize(config_path="../../configs", version_base="1.2"):
        config = compose(
            config_name="config",
            overrides=["data=turbulence", "training=turbulence_optuna"],
        )
    return config


main_experiment_folder = "corrector/models/fno_turbulence"
experiment_path = main_experiment_folder


def energy_calculation_plotting(config):
    dataset_turb = dataset(scenarios_to_use=config.data.scenarios, cfg_data=config.data)

    config.models.hidden_channels = 13
    config.models.n_fourier_layers = 2
    config.models.fourier_modes = 14
    config.models.shifting_modes = 1

    model_cfg = OmegaConf.to_container(config.models, resolve=True)
    model_name = model_cfg.pop("_name_", None)

    key = jax.random.PRNGKey(config.training.rng_key)

    # ─── New Fields ───────────────────────────────────────────────────────
    model_cfg["postprocessing_floor"] = True
    # model_cfg["output_channels"] = 5

    model = instantiate(model_cfg, key=key)
    model = eqx.tree_deserialise_leaves(
        os.path.join(
            main_experiment_folder,
            "fno.eqx",
        ),
        model,
    )
    # model = eqx.tree_at(lambda m: m.postprocessing_floor, model, False)
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    trainable_params = sum(
        x.size
        for x in jax.tree_util.tree_leaves(eqx.filter(neural_net_params, eqx.is_array))
    )
    print(
        f" ✅ Initialized model '{model_name}' successfully with # of params {trainable_params}"
    )
    corrector_config = CorrectorConfig(
        corrector=True,
        network_static=neural_net_static,
        correct_from_beggining=False,
        start_correction_time=0.06,
    )
    corrector_params = CorrectorParams(network_params=neural_net_params)

    (
        sim_bundle_hr,
        sim_bundle_lr,
        hr_snapshot_data,
        lr_snapshot_data,
        lr_ml_snapshot_data,
    ) = dataset_turb.hr_lr_ml_states_integration(
        corrector_config=corrector_config, corrector_params=corrector_params
    )
    # state, helper_data, gamma, config, registered_variables
    v_internal_energy = jax.vmap(
        calculate_internal_energy, in_axes=(0, None, None, None, None)
    )
    v_kinetic_energy = jax.vmap(calculate_kinetic_energy, in_axes=(0, None, None, None))
    v_total_energy = jax.vmap(
        calculate_total_energy, in_axes=(0, None, None, None, None, None)
    )
    energies = []
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for snapshot_data, sim_bundle, name, color in zip(
        [hr_snapshot_data, lr_snapshot_data, lr_ml_snapshot_data],
        [sim_bundle_hr, sim_bundle_lr, sim_bundle_lr],
        [
            f"{str(config.data.hr_res)}",
            f"{str(config.data.hr_res // config.data.downscaling_factor)}",
            f"{str(config.data.hr_res // config.data.downscaling_factor)} corrected",
        ],
        colors,
    ):
        internal_energies = v_internal_energy(
            snapshot_data.states,
            sim_bundle.helper,
            sim_bundle.params.gamma,
            sim_bundle.config,
            sim_bundle.reg_vars,
        )
        kinetic_energies = v_kinetic_energy(
            snapshot_data.states,
            sim_bundle.helper,
            sim_bundle.config,
            sim_bundle.reg_vars,
        )
        total_energies = v_total_energy(
            snapshot_data.states,
            sim_bundle.helper,
            sim_bundle.params.gamma,
            sim_bundle.params.gravitational_constant,
            sim_bundle.config,
            sim_bundle.reg_vars,
        )

        energies.append(
            {
                "name": name,
                "color": color,
                "internal_e": internal_energies,
                "kinetic_e": kinetic_energies,
                "total_e": total_energies,
                "times": snapshot_data.time_points,
            }
        )
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for energy in energies:
        c = energy["color"]

        # Internal energy
        ax[0].plot(
            energy["times"],
            energy["internal_e"],
            label=f"{energy['name']}",
            color=c,
        )
        # ax[0].plot(
        #     energy["times"],
        #     energy["internal_e"][0] - energy["internal_e"],
        #     linestyle="dashed",
        #     color=c,
        # )

        # Kinetic energy
        ax[1].plot(
            energy["times"],
            energy["kinetic_e"],
            label=f"{energy['name']}",
            color=c,
        )
        # ax[1].plot(
        #     energy["times"],
        #     energy["kinetic_e"][0] - energy["kinetic_e"],
        #     linestyle="dashed",
        #     color=c,
        # )

        # Total energy
        ax[2].plot(
            energy["times"],
            energy["total_e"],
            label=f"{energy['name']}",
            color=c,
        )
        ax[2].plot(
            energy["times"],
            energy["total_e"][0] - energy["total_e"],
            linestyle="dashed",
            color=c,
            label=f"{energy['name']} difference with initial",
        )

    # ---- Titles, labels, legend ----
    ax[0].set_title("Internal Energy")
    ax[1].set_title("Kinetic Energy")
    ax[2].set_title("Total Energy")

    for a in ax:
        a.legend()
        a.set_xlabel("Time")
        a.set_ylabel("Energy")
    plt.savefig("corrector/figures/energy_time_evolution.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = load_config()
    energy_calculation_plotting(config=config)
