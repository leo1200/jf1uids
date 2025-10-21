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


from corrector_src.training.sol_one_training_snapshots import time_integration_train
from jf1uids import time_integration
from corrector_src.training.training_config import TrainingConfig, TrainingParams
from corrector_src.model.cnn_mhd_model import CorrectorCNN
from corrector_src.model._cnn_mhd_corrector_options import (
    CNNMHDParams,
    CNNMHDconfig,
    CorrectorConfig,
    CorrectorParams,
)
from corrector_src.training.loss import mse_loss
from corrector_src.data.load_sim import (
    load_states,
    integrate_blast,
    filepath_state,
    prepare_initial_state,
)
from corrector_src.data.dataset import dataset

import equinox as eqx
import matplotlib.pyplot as plt
import optax
import time
import hydra
from hydra.utils import get_original_cwd
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def training_loop(cfg):
    print("Hydra run dir:", os.getcwd())
    epochs = cfg.training.epochs
    n_look_behind = cfg.training.n_look_behind
    loss_timesteps = jnp.array(cfg.data.snapshot_timepoints)
    assert loss_timesteps.any() > cfg.data.t_end, (
        "found value greater than the end time in the snapshot timepoints"
    )
    # Training configuration
    training_config = TrainingConfig(
        # compute_intermediate_losses=True,
        # n_look_behind=n_look_behind,
        # loss_weights=None,
        # use_relative_error=False,
    )
    training_params = TrainingParams(loss_calculation_times=loss_timesteps)
    loss_function = mse_loss

    # Initialize model and optimizer
    key = jax.random.PRNGKey(cfg.training.rng_key)
    model = instantiate(cfg.models, key=key)

    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    corrector_config = CorrectorConfig(corrector=True, network_static=neural_net_static)
    corrector_params = CorrectorParams(network_params=neural_net_params)
    optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adam(float(cfg.training.learning_rate)),
    )
    opt_state = optimizer.init(neural_net_params)

    snapshot_losses = []
    epoch_losses = []
    rng_seed = 112
    # assert cfg.data.precomputed_data != cfg.data.generate_data_on_fly, (
    #     f"given use precomputed data {cfg.data.precomputed_data} and data on the fly {cfg.data.generate_data_on_fly}"
    # )
    # if cfg.data.precomputed_data and not cfg.data.generate_data_on_fly:
    #     filepath = filepath_state(
    #         [
    #             get_original_cwd(),
    #             "corrector_src",
    #             "data/states",
    #             f"ground_truth_{cfg.data.num_checkpoints}.npy",
    #         ]
    #     )
    #     if not os.path.exists(filepath):
    #         gt_cfg_data = cfg.data
    #         gt_cfg_data.debug = False
    #         ground_truth, _ = integrate_blast(
    #             cfg.data, filepath, rng_seed, downscale=True, save_file=True
    #         )
    #     else:
    #         ground_truth = load_states(filepath)

    gt_cfg_data = cfg.data
    gt_cfg_data.debug = False
    dataset_creator = dataset(gt_cfg_data.scenarios, gt_cfg_data)

    if cfg.training.early_stopping:
        best_loss = float("inf")
        best_params = neural_net_params

    if not cfg.data.generate_data_on_fly and len(gt_cfg_data.scenarios) == 1:
        # ground_truth, _ = integrate_blast(cfg.data, None, rng_seed, downscale=True)
        (
            ground_truth,
            (
                initial_state_lr,
                initial_config_lr,
                initial_params_lr,
                initial_helper_data_lr,
                initial_registered_variables_lr,
            ),
        ) = dataset_creator.train_initializator(
            downscale=cfg.data.downscaling_factor, rng_seed=rng_seed
        )
        print("gt_shape:", jnp.shape(ground_truth))
        initial_config_lr = initial_config_lr._replace(
            corrector_config=corrector_config
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
            ) = dataset_creator.train_initializator(
                resolution=cfg.data.hr_res,
                downscale=cfg.data.downscaling_factor,
                rng_seed=None,
                # scenario selection if needed
                corrector_config=corrector_config,
                corrector_params=corrector_params,
            )
        else:
            initial_params_lr = initial_params_lr._replace(
                corrector_params=corrector_params
            )
        time_train = time.time()
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

        time_train = time.time() - time_train

        if np.isnan(np.mean(losses)):
            print("nan found in loss, stopping the training")
            break
        snapshot_losses.append(losses.flatten())
        epoch_losses.append(np.mean(losses))
        corrector_params = corrector_params._replace(network_params=new_network_params)
        print(f"epoch {i} time_train {time_train} loss {epoch_losses[-1]}")

    snapshot_losses = np.array(snapshot_losses)
    snapshot_losses = snapshot_losses.flatten()

    plt.figure()
    plt.plot(snapshot_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig("snapshot_loss_curve.png")
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

    final_model = eqx.combine(new_network_params, neural_net_static)
    eqx.tree_serialise_leaves("cnn_model.eqx", final_model)
    np.savez(
        "losses.npz",
        n_look_behind=np.array(n_look_behind),
        epoch_losses=np.array(epoch_losses),
        snapshot_losses=np.array(snapshot_losses),
    )


if __name__ == "__main__":
    training_loop()
