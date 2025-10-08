# from autocvd import autocvd

# autocvd(num_gpus=1)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import time_integration
from jf1uids.data_classes.simulation_helper_data import HelperData, get_helper_data

import corrector_src.data.blast_creation as blast
from corrector_src.utils.downaverage import downaverage_states

import numpy as np

import jax.numpy as jnp
import jax

jax.log_compiles(True)

from corrector_src.training.sol_one_training_snapshots import time_integration

from corrector_src.training.training_config import TrainingConfig, TrainingParams
from corrector_src.model._cnn_mhd_corrector import CorrectorCNN
from corrector_src.model._cnn_mhd_corrector_options import (
    CNNMHDParams,
    CNNMHDconfig,
)
from corrector_src.training.loss import mse_loss
from corrector_src.data.load_sim import (
    load_states,
    integrate_blast,
    filepath_state,
    prepare_initial_state,
)

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
        compute_intermediate_losses=True,
        n_look_behind=n_look_behind,
        loss_weights=None,
        use_relative_error=False,
    )
    training_params = TrainingParams(loss_calculation_times=loss_timesteps)
    loss_function = mse_loss

    # Initialize model and optimizer
    key = jax.random.PRNGKey(cfg.training.rng_key)
    model = instantiate(cfg.models, key=key)

    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True, network_static=neural_net_static
    )
    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)
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
    ground_truth, _ = integrate_blast(cfg.data, None, rng_seed, downscale=True)
    print("gt_shape:", jnp.shape(ground_truth))

    for i in range(epochs):
        # if cfg.data.generate_data_on_fly:
        #     ground_truth, rng_seed = integrate_blast(cfg.data, downscale=True)

        initial_state, config, params, helper_data, registered_variables = (
            prepare_initial_state(
                cfg_data=cfg.data,
                rng_seed=rng_seed,
                cnn_mhd_corrector_config=cnn_mhd_corrector_config,
                cnn_mhd_corrector_params=cnn_mhd_corrector_params,
                downscale=True,
            )
        )

        time_train = time.time()
        losses, new_network_params, opt_state, _ = time_integration(
            primitive_state=initial_state,
            config=config,
            params=params,
            helper_data=helper_data,
            registered_variables=registered_variables,
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
        cnn_mhd_corrector_params = cnn_mhd_corrector_params._replace(
            network_params=new_network_params
        )
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
