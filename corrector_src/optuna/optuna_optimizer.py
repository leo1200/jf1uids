from autocvd import autocvd

autocvd(num_gpus=1)

from corrector_src.model._corrector_options import CorrectorConfig
from corrector_src.optuna.optuna_train_model import train_model

from hydra.utils import instantiate
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CorrectorParams,
)
from optuna import trial
import optuna
import os
from hydra import initialize, compose
from omegaconf import OmegaConf
from corrector_src.loss.sgs_turb_loss import make_loss_function
import h5py
from corrector_src.data.dataset import dataset, SimulationBundle
from jf1uids.time_stepping import time_integration
from jaxtyping import PyTree
from jax import vmap, clear_caches
import jax.numpy as jnp
import gc
from functools import partial

n_trials = 20
study_name = "turbulence"
experiment_folder = ""


def main():
    with initialize(config_path="../../configs", version_base="1.2"):
        training_cfg = compose(
            config_name="config",
            overrides=["data=turbulence", "training=turbulence_optuna"],
        )
        validation_cfg = compose(
            config_name="config",
            overrides=["data=turbulence_dataset", "training=turbulence_optuna"],
        )

    optuna_optimize(training_cfg, validation_cfg)


def optuna_optimize(training_config: OmegaConf, validation_config: OmegaConf):
    experiment_folder = os.path.abspath(
        "/export/home/jalegria/Thesis/jf1uids/corrector/optuna_studies"
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{os.path.join(experiment_folder, 'turbulence.db')}",
        load_if_exists=True,
        directions=["minimize", "minimize", "minimize"],
    )
    study.optimize(
        partial(
            objective,
            training_config=training_config,
            validation_config=validation_config,
        ),
        show_progress_bar=True,
        n_trials=n_trials,
        gc_after_trial=True,
    )


def objective(
    trial: trial.Trial, training_config: OmegaConf, validation_config: OmegaConf
):
    params = {
        "models": {
            "hidden_channels": trial.suggest_int("hidden_channels", 10, 30),
            "n_fourier_layers": trial.suggest_int("n_fourier_layers", 1, 5),
            "fourier_modes": trial.suggest_int("fourier_modes", 4, 20),
            "shifting_modes": trial.suggest_int("shifting_modes", 0, 10),
        },
        "training": {
            "mse_loss": trial.suggest_float("mse_loss", 0, 1),
            "spectral_energy_loss": trial.suggest_float("spectral_energy_loss", 0, 1),
            "rate_of_strain_loss": trial.suggest_float("rate_of_strain_loss", 0, 1),
        },
        "data": {},
    }
    print(params)
    for config_name, config_overrides in params.items():
        for parameter, value in config_overrides.items():
            training_config[config_name][parameter] = value

    try:
        network_params, static_params = train_model(training_config)

    except ValueError as e:
        if "NaN" in str(e):
            raise optuna.TrialPruned()
        elif "Out of memory" in str(e):
            raise optuna.TrialPruned()
        else:
            raise e

    losses = eval_model(
        network_params=network_params,
        static_params=static_params,
        cfg_training=validation_config.training,
        cfg_data=validation_config.data,
    )
    clear_caches()
    gc.collect()
    return losses


def eval_model(
    network_params: PyTree,
    static_params: PyTree,
    cfg_training: OmegaConf,
    cfg_data: OmegaConf,
):
    sim_bundle_creator = dataset(scenarios_to_use=cfg_data.scenarios, cfg_data=cfg_data)
    local_validation_bundle = sim_bundle_creator.sim_initializator(
        resolution=cfg_data.hr_res // cfg_data.downscaling_factor,
        corrector_config=CorrectorConfig(corrector=True, network_static=static_params),
        corrector_params=CorrectorParams(network_params=network_params),
    )

    cfg_temp = OmegaConf.merge(
        cfg_training,
        {"mse_loss": 1, "rate_of_strain_loss": 1, "spectral_energy_loss": 1},
    )
    loss_fn, *_ = make_loss_function(cfg_training=cfg_temp)
    v_loss_fn = vmap(loss_fn, in_axes=(0, 0, None, None, None))
    with h5py.File(
        os.path.abspath(
            "/export/data/jalegria/jf1uids_turbulence_sol/validation_data_turbulence.h5"
        ),
        "r",
    ) as h5file:
        gt_states = h5file["gt_states"]
        initial_states = h5file["initial_state"]

        accumulated_components = {}

        for i, initial_state in enumerate(initial_states):
            local_validation_bundle.initial_state = jnp.array(
                initial_state, dtype="float64"
            )
            snapshot_data = time_integration(
                **local_validation_bundle.unpack_integrate()
            )
            loss, components = v_loss_fn(
                snapshot_data.states,
                gt_states[i],
                local_validation_bundle.config,
                local_validation_bundle.reg_vars,
                local_validation_bundle.params,
            )

            for name, val in components.items():
                accumulated_components[name] = accumulated_components.get(
                    name, 0.0
                ) + float(jnp.sum(val))
    # energies = h5file["first_snapshot_energy"]
    # masses = h5file["first_snapshot_mass"]
    # seeds = h5file["seed"]

    mean_components = {
        name: val / len(initial_states) for name, val in accumulated_components.items()
    }

    return (
        float(mean_components["mse"]) / cfg_training.mse_weight,
        float(mean_components["strain"]) / cfg_training.strain_weight,
        float(mean_components["spectral"]) / cfg_training.spectral_weight,
    )


if __name__ == "__main__":
    main()
