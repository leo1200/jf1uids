from autocvd import autocvd

autocvd(num_gpus=1)

import os

# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import time_integration

import corrector_src.data.blast_creation as blast
from corrector_src.utils.downaverage import downaverage_states

import numpy as np

import jax.numpy as jnp
import jax


from corrector_src.training.solver_in_loop_train_tbt import time_integration_training_tbt

from corrector_src.training.training_config import TrainingConfig
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

    assert cfg.data.num_snapshots % n_look_behind == 0, (
        f"total steps not divisible by data lag, got {n_look_behind} and {cfg.data.num_snapshots}"
    )

    # Training configuration
    training_config = TrainingConfig(
        compute_intermediate_losses=True,
        n_look_behind=n_look_behind,
        loss_weights=None,
        use_relative_error=False,
    )
    loss_function = mse_loss

    key = jax.random.PRNGKey(cfg.training.rng_key)
    model = instantiate(cfg.models, key=key)

    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True, network_static=neural_net_static
    )
    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(float(cfg.training.learning_rate)),
    )
    opt_state = optimizer.init(neural_net_params)

    snapshot_losses = []
    epoch_losses = []
    rng_seed = None
    for i in range(epochs):
        initial_state_lr, config_lr, params, helper_data_lr, registered_variables = (
            prepare_initial_state(
                cfg.data,
                rng_seed,
                cnn_mhd_corrector_config,
                cnn_mhd_corrector_params,
                downscale=True
            )
        )

        initial_state_hr, config_hr, params_hr, helper_data_hr, registered_variables = (
            prepare_initial_state(
                cfg.data,
                rng_seed,
                None,
                None,
                downscale=False
            )
        )

        time_train = time.time()
        if cfg.training.debug:
            err, (losses, new_network_params, opt_state) = time_integration_training_tbt(
                initial_state_hr=initial_state_hr,
                config_hr=config_hr,
                config_lr=config_lr,
                params=params,
                helper_data_hr=helper_data_hr,
                helper_data_lr=helper_data_lr,
                registered_variables=registered_variables,
                training_config=training_config,
                optimizer=optimizer,
                opt_state=opt_state,
                loss_function=loss_function,
                save_full_sim=False,
            )

            err.throw()
        else:
            losses, new_network_params, opt_state = time_integration_training_tbt(
                initial_state_hr=initial_state_hr,
                config_hr=config_hr,
                config_lr=config_lr,
                params=params,
                helper_data_hr=helper_data_hr,
                helper_data_lr=helper_data_lr,
                registered_variables=registered_variables,
                training_config=training_config,
                optimizer=optimizer,
                opt_state=opt_state,
                loss_function=loss_function,
                save_full_sim=False,
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
