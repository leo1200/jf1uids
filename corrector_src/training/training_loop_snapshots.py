from autocvd import autocvd

autocvd(num_gpus=1)

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

import numpy as np
import jax.numpy as jnp
import jax

# corrector_src
from corrector_src.training.sol_one_training_snapshots import (
    time_integration as time_integration_train,
)
from corrector_src.training.training_config import TrainingConfig, TrainingParams
from corrector_src.model.cnn_mhd_model import CorrectorCNN
from corrector_src.model.fno_hd_force_corrector import TurbulenceSGSForceCorrectorFNO
from corrector_src.model._cnn_mhd_corrector_options import (
    # CNNMHDParams,
    # CNNMHDconfig,
    CorrectorConfig,
    CorrectorParams,
)
import corrector_src.loss.sgs_turb_loss as loss
from corrector_src.data.load_sim import (
    load_states,
    integrate_blast,
    filepath_state,
    prepare_initial_state,
)
from corrector_src.data.dataset import dataset
from corrector_src.training.early_stopper import EarlyStopper
from corrector_src.utils.printing_config_summary import print_data_config_summary

# jf1uids
from jf1uids import time_integration
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams

# other stuff
import equinox as eqx
import matplotlib.pyplot as plt
import optax
import time
import hydra
from hydra.utils import get_original_cwd
from hydra.utils import instantiate
from omegaconf import OmegaConf
from functools import partial
from typing import Tuple


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def training_loop(cfg):
    print("Hydra run dir:", os.getcwd())
    print_data_config_summary(cfg.data)
    epochs = cfg.training.epochs
    n_look_behind = cfg.training.n_look_behind
    loss_timesteps = jnp.array(cfg.data.snapshot_timepoints)
    assert not (loss_timesteps > cfg.data.t_end).any(), (
        "found value greater than the end time in the snapshot timepoints"
    )

    # ─── Training Configuration ───────────────────────────────────────────

    training_config = TrainingConfig()
    training_params = TrainingParams(loss_calculation_times=loss_timesteps)

    def make_loss_function(cfg_training):
        """Builds a pure JAX-compatible loss function using values from cfg_training."""

        # Define loss components as a dict of {name: (weight, fn)}
        loss_fns = {
            "mse": (
                cfg_training["mse_loss"],
                lambda pred, gt, config, registered_vars, params: loss.mse_loss(
                    pred, gt
                ),
            ),
            "strain": (
                cfg_training["rate_of_strain_loss"],
                lambda pred,
                gt,
                config,
                registered_vars,
                params: loss.rate_of_strain_loss(pred, gt, config, registered_vars),
            ),
            # "spectral": (
            #     cfg_training["spectral_energy_loss"],
            #     lambda pred,
            #     gt,
            #     config,
            #     registered_vars,
            #     params: loss.spectral_energy_loss(
            #         pred, gt, config, registered_vars, params
            #     ),
            # ),
        }
        active_loss_indices = {
            i: name.replace("_loss", "")
            for i, (name, (w, _)) in enumerate(loss_fns.items())
            if w != 0
        }

        @partial(jax.jit, static_argnames=["config", "registered_variables"])
        def loss_function(
            predicted, ground_truth, config, registered_variables, params
        ):
            total = 0.0
            components = {}

            for name, (weight, fn) in loss_fns.items():
                if weight > 0:
                    val = fn(
                        predicted, ground_truth, config, registered_variables, params
                    )
                    components[name] = val
                    total += weight * val

            for name, value in components.items():
                jax.debug.print("{name}: {value}", name=name, value=value)

            return total, components

        return loss_function, active_loss_indices

    loss_function, active_loss_indices = make_loss_function(cfg.training)

    # ─── Creating Model And Optimizer ─────────────────────────────────────

    model_cfg = OmegaConf.to_container(cfg.models, resolve=True)
    model_name = model_cfg.pop("_name_", None)

    key = jax.random.PRNGKey(cfg.training.rng_key)
    model = instantiate(model_cfg, key=key)

    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    trainable_params = sum(
        x.size
        for x in jax.tree_util.tree_leaves(eqx.filter(neural_net_params, eqx.is_array))
    )
    print(
        f" ✅ Initialized model '{model_name}' successfully with # of params {trainable_params}"
    )

    corrector_config = CorrectorConfig(corrector=True, network_static=neural_net_static)
    corrector_params = CorrectorParams(network_params=neural_net_params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(float(cfg.training.learning_rate)),
    )
    opt_state = optimizer.init(neural_net_params)

    snapshot_losses = []
    epoch_losses = []

    gt_cfg_data = cfg.data
    # gt_cfg_data.debug = False
    dataset_creator = dataset(gt_cfg_data.scenarios, gt_cfg_data)
    print(f" ✅ Using data {dataset_creator.scenario_list}")

    # ─── Early Stopping ───────────────────────────────────────────────────

    if cfg.training.early_stopping:
        patience = 5
        print(f" 🥱 Using early stopper with patience {patience}")
        early_stopper = EarlyStopper(patience=5)
        best_params = neural_net_params
    else:
        early_stopper = None

    if not cfg.data.generate_data_on_fly and len(gt_cfg_data.scenarios) == 1:
        (
            ground_truth,
            (
                initial_state_lr,
                initial_config_lr,
                initial_params_lr,
                initial_helper_data_lr,
                initial_registered_variables_lr,
            ),
        ) = creating_data(
            dataset=dataset_creator,
            cfg=cfg,
            corrector_config=corrector_config,
            corrector_params=corrector_params,
        )
    # def validate_model(network_params, cfg_data, corrector_config, corrector_params):
    #     cfg_data.use_specific_snapshot_timepoints = True
    #     cfg_data.snapshot_timepoints = np.linspace(0.0, cfg_data.t_end, 100).tolist()
    #     ground_truth, lr_sim = dataset_creator.train_initializator(
    #         resolution=cfg.data.hr_res,
    #         downscale=cfg.data.downscaling_factor,
    #         rng_seed=None,
    #         # scenario selection if needed
    #         corrector_config=corrector_config,
    #         corrector_params=corrector_params,
    #     )
    early_stop = False
    for i in range(epochs):
        if cfg.data.generate_data_on_fly:
            (
                ground_truth,
                (
                    initial_state_lr,
                    initial_config_lr,
                    initial_params_lr,
                    initial_helper_data_lr,
                    initial_registered_variables_lr,
                ),
            ) = creating_data(
                dataset=dataset_creator,
                cfg=cfg,
                corrector_config=corrector_config,
                corrector_params=corrector_params,
            )
        else:
            initial_params_lr = initial_params_lr._replace(
                corrector_params=corrector_params
            )
        time_train = time.time()
        initial_config_lr = initial_config_lr._replace(return_snapshots=False)
        losses, new_network_params, opt_state, _ = time_integration_train(
            primitive_state=initial_state_lr,
            config=initial_config_lr,
            params=initial_params_lr,
            helper_data=initial_helper_data_lr,
            registered_variables=initial_registered_variables_lr,
            optimizer=optimizer,
            loss_function=loss_function,
            opt_state=opt_state,
            target_data=ground_truth,
            training_config=training_config,
            training_params=training_params,
        )
        epoch_loss = np.mean(losses)
        time_train = time.time() - time_train

        if np.isnan(np.mean(losses)):
            print("nan found in loss, stopping the training")
            break

        snapshot_losses.append(losses.flatten())
        epoch_losses.append(epoch_loss)

        if early_stopper is not None:
            early_stop = early_stopper.early_stop(epoch_loss)
            if epoch_loss < early_stopper.min_validation_loss:
                best_params = new_network_params
            if early_stop:
                print("🚨 Early stopping 🚨")
                break

        corrector_params = corrector_params._replace(network_params=new_network_params)
        print(f" 🟡 Epoch {i} time_train {time_train:2f} loss {epoch_losses[-1]:2f}")

    snapshot_losses = np.array(snapshot_losses)
    snapshot_losses = snapshot_losses.flatten()

    plt.figure()
    # plt.plot(snapshot_losses[:, 0], label="Train Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Losses")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("snapshot_loss_curve.png")
    # plt.close()

    for i, name in active_loss_indices:
        plt.plot(snapshot_losses[:, i], label=name)

    # Optionally, plot total loss if you’re storing it as the last column

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig("components_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epoch_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig("epoch_loss_curve.png")
    plt.close()

    if early_stopper is not None:
        final_model = eqx.combine(best_params, neural_net_static)
    else:
        final_model = eqx.combine(new_network_params, neural_net_static)

    eqx.tree_serialise_leaves(f"{model_name}.eqx", final_model)
    np.savez(
        "losses.npz",
        n_look_behind=np.array(n_look_behind),
        epoch_losses=np.array(epoch_losses),
        snapshot_losses=np.array(snapshot_losses),
    )


def creating_data(
    dataset: dataset,
    cfg: OmegaConf,
    corrector_config: CorrectorConfig,
    corrector_params: CorrectorParams,
) -> Tuple[
    jnp.ndarray,
    Tuple[
        np.ndarray,
        SimulationConfig,
        SimulationParams,
        HelperData,
        RegisteredVariables,
    ],
]:
    "creates data and makes sure that the ground truth hr and lr dont have any nans"
    is_nan_data = True
    while is_nan_data:
        time_start = time.time()
        (
            ground_truth,
            (
                initial_state_lr,
                initial_config_lr,
                initial_params_lr,
                initial_helper_data_lr,
                initial_registered_variables_lr,
            ),
        ) = dataset.train_initializator(
            resolution=cfg.data.hr_res,
            downscale=cfg.data.downscaling_factor,
            rng_seed=None,
            # scenario selection if needed
            corrector_config=corrector_config,
            corrector_params=corrector_params,
        )
        time_hr = time.time()
        is_nan_data = jnp.any(jnp.isnan(ground_truth))
        if is_nan_data:
            print("found nan in hr_data repeating the calculation")
        else:
            not_ml_config_lr = initial_config_lr._replace(
                return_snapshots=True, corrector_config=CorrectorConfig(corrector=False)
            )

            final_states_lr = time_integration(
                initial_state_lr,
                not_ml_config_lr,
                initial_params_lr,
                initial_helper_data_lr,
                initial_registered_variables_lr,
            )
            time_lr = time.time()
            is_nan_data = jnp.any(jnp.isnan(final_states_lr.states))
            if is_nan_data:
                print("nan found in lr without ml, getting another initial state")
        print(
            f"Time taken to create data hr {time_hr - time_start}, lr {time_lr - time_hr}"
        )
    return (
        ground_truth,
        (
            initial_state_lr,
            initial_config_lr,
            initial_params_lr,
            initial_helper_data_lr,
            initial_registered_variables_lr,
        ),
    )


if __name__ == "__main__":
    training_loop()
