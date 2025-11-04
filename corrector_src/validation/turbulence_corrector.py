from autocvd import autocvd

autocvd(num_gpus=1)

import os
import numpy as np
import jax.numpy as jnp
import jax


# corrector_src
from corrector_src.model.cnn_mhd_model import CorrectorCNN
from corrector_src.model.fno_hd_force_corrector import TurbulenceSGSForceCorrectorFNO
from corrector_src.model._cnn_mhd_corrector_options import (
    # CNNMHDParams,
    # CNNMHDconfig,
    CorrectorConfig,
    CorrectorParams,
)
from corrector_src.loss.sgs_turb_loss import get_energy, make_loss_function
from corrector_src.utils.power_spectra_1d import pk_jax_1d
from corrector_src.data.dataset import dataset
from corrector_src.utils.downaverage import downaverage_states

# jf1uids
from jf1uids import time_integration
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams

# other stuff
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib import animation
import optax
import time
import hydra
from hydra.utils import get_original_cwd
from hydra.utils import instantiate
from omegaconf import OmegaConf
from functools import partial
from typing import Tuple
import Pk_library as PKL

main_experiment_folder = "../../experiments/turbulence_force_corrector/"
experiment_name = "2025-11-03_17-35-49"
experiment_path = os.path.join(main_experiment_folder, experiment_name)
config_path = os.path.join(experiment_path, ".hydra")
animation_name = "corrector/figures/energy_spectra"
figure_name = "corrector/figures/loss_components"
pk_comparison = False


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def turb_model_validation(cfg):
    rng_seed = 448505923
    num_snapshots = 30
    use_specific_snapshot_timepoints = True
    specific_snapshots = np.arange(0.0, cfg.data.t_end, cfg.data.t_end / 30).tolist()
    if cfg.data.t_end not in specific_snapshots:
        specific_snapshots.append(cfg.data.t_end)
    cfg.data.use_specific_snapshot_timepoints = use_specific_snapshot_timepoints
    cfg.data.num_snapshots = num_snapshots
    cfg.data.return_snapshots = True
    cfg.data.snapshot_timepoints = specific_snapshots
    # cfg.data.differentiation_mode = 0  # FOWARDS

    cfg.training.spectral_energy_loss = 1.0

    model_cfg = OmegaConf.to_container(cfg.models, resolve=True)
    model_name = model_cfg.pop("_name_", None)
    key = jax.random.PRNGKey(cfg.training.rng_key)
    model = instantiate(model_cfg, key=key)
    model = eqx.tree_deserialise_leaves(
        os.path.join(
            "/export/home/jalegria/Thesis/jf1uids/experiments/turbulence_force_corrector",
            experiment_name,
            "fno.eqx",
        ),
        model,
    )

    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    trainable_params = sum(
        x.size
        for x in jax.tree_util.tree_leaves(eqx.filter(neural_net_params, eqx.is_array))
    )
    print(
        f" ✅ Initialized model '{model_name}' successfully with # of params {trainable_params}"
    )
    corrector_config = CorrectorConfig(corrector=True, network_static=neural_net_static)
    corrector_params = CorrectorParams(neural_net_params)

    dataset_turb = dataset([1], cfg.data)

    is_nan_data = True
    jit_time_integration = jax.jit(time_integration)
    while is_nan_data:
        try:
            (
                (
                    initial_state_hr,
                    config_hr,
                    params_hr,
                    helper_data_hr,
                    registered_variables_hr,
                ),
                (
                    initial_state_lr,
                    config_lr,
                    params_lr,
                    helper_data_lr,
                    registered_variables_lr,
                ),
            ) = dataset_turb.hr_lr_initializator(
                resolution=cfg.data.hr_res,
                downscale=cfg.data.downscaling_factor,
                rng_seed=rng_seed,
                # corrector_config=corrector_config,
                # corrector_params=corrector_params,
            )
            start_hr = time.time()
            hr_states = jit_time_integration(
                initial_state_hr,
                config_hr,
                params_hr,
                helper_data_hr,
                registered_variables_hr,
            ).states
            time_hr = time.time() - start_hr
            # print(jnp.any(jnp.isnan(hr_states.states)))
            start_lr = time.time()
            states_lr = jit_time_integration(
                initial_state_lr,
                config_lr,
                params_lr,
                helper_data_lr,
                registered_variables_lr,
            ).states
            time_lr = time.time() - start_lr
            is_nan_data = False
            print("data created")
        except RuntimeError as e:
            if "float NaN" in str(e):
                print("nan founds in data trying another seed")
                rng_seed = None
                is_nan_data = True

    config_lr_sol = config_lr._replace(corrector_config=corrector_config)
    params_lr_sol = params_lr._replace(corrector_params=corrector_params)
    start_time_lr_sol = time.time()
    states_lr_sol = jit_time_integration(
        initial_state_lr,
        config_lr_sol,
        params_lr_sol,
        helper_data_lr,
        registered_variables_lr,
    ).states
    time_lr_sol = time.time() - start_time_lr_sol
    loss_fn, loss_from_components, active_loss_indices = make_loss_function(
        cfg.training
    )
    downscaled_hr_states = downaverage_states(hr_states, cfg.data.downscaling_factor)
    vloss = jax.vmap(loss_fn, in_axes=(0, 0, None, None, None), out_axes=0)
    total_loss, loss_components = vloss(
        states_lr_sol,
        downscaled_hr_states,
        config_lr,
        registered_variables_lr,
        params_lr,
    )

    plt.figure()

    for i, (name, weight) in active_loss_indices.items():
        plt.plot(
            loss_components[name] / jnp.linalg.norm(loss_components[name]), label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Normalized Training Loss Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()

    spectral_loss = loss_components["spectral"]
    vget_energy = jax.vmap(get_energy, in_axes=(0, None, None, None))
    labels = ["low res sol", "high res", "low res"]
    states_list = (states_lr_sol, downscaled_hr_states, states_lr)
    energies = {}
    for states, label in zip(states_list, labels):
        energies[label] = vget_energy(
            states, config_lr, registered_variables_lr, params_lr
        )

    vpower = jax.vmap(pk_jax_1d, in_axes=(0, None, None))
    spectrums = []
    for label, energy in energies.items():
        k, Pk, Nmodes = vpower(energy, 1.0, 0)

        spectrums.append(
            {
                "spectrum": Pk,
                "k": k,
                "label": label,
                "time_points": specific_snapshots,
            }
        )
        spectrums_pkl = []
        k_pkl = []
        if pk_comparison:
            for e in energy:
                pk_energy = PKL.Pk(
                    delta=np.array(e, dtype=np.float32),
                    BoxSize=1,
                    axis=0,
                    MAS="None",
                    threads=6,
                    verbose=False,
                )
                spectrums_pkl.append(pk_energy.Pk1D)
                k_pkl.append(pk_energy.k1D)
            spectrums.append(
                {
                    "spectrum": spectrums_pkl,
                    "k": k_pkl,
                    "label": label + "_pkl",
                    "time_points": specific_snapshots,
                }
            )

    fig, ax = plt.subplots(figsize=(8, 6))

    lines = []
    for spec in spectrums:
        (line,) = ax.plot([], [], lw=2, label=spec["label"])
        lines.append(line)

    # --- k^-2 reference line ---
    k_ref = spectrums[0]["k"][0]  # use k from first resolution
    P_ref = k_ref**-2
    (ref_line,) = ax.plot(k_ref, P_ref, "k--", label=r"$k^{-2}$ reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.set_ylim(1e-6, 4e-1)
    ax.set_title("Power spectra for all resolutions")
    ax.legend()
    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")

    # --- Animation setup ---
    def init():
        for line in lines:
            line.set_data([], [])
        title.set_text(
            f"t = {spectrums[0]['time_points'][0]:.2f} loss = {spectral_loss[0]:.2f}"
        )
        return lines + [title]

    def animate(j):
        for line, spec in zip(lines, spectrums):
            line.set_data(spec["k"][j], spec["spectrum"][j])
        title.set_text(
            f"t = {spectrums[0]['time_points'][j]:.2f}  loss = {spectral_loss[j]:.2f}"
        )
        return lines + [title]

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(spectrums[0]["time_points"]),
        interval=100,
        blit=False,
    )

    ani.save(
        animation_name + ".mp4", writer=animation.FFMpegWriter(fps=10, bitrate=1800)
    )
    plt.show()

    print("=" * 30)
    print(f"High res time for {cfg.data.hr_res} resolution : {time_hr}")
    print(
        f"Low res time for {cfg.data.hr_res // cfg.data.downscaling_factor} resolution : {time_lr}"
    )
    print(
        f"Low res time for {cfg.data.hr_res // cfg.data.downscaling_factor} resolution and solver in the loop with {trainable_params} parameters: {time_lr_sol}"
    )
    print("=" * 30)


if __name__ == "__main__":
    turb_model_validation()
