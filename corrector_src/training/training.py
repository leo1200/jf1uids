from autocvd import autocvd

autocvd(num_gpus=1)

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

from jf1uids.option_classes.simulation_config import finalize_config
from jf1uids import time_integration

import corrector_src.data.blast_creation as blast
from corrector_src.utils.downaverage import downaverage_states

import numpy as np

import jax.numpy as jnp
import jax

from corrector_src.training.time_integration_w_training import (
    time_integration_w_training,
)


from corrector_src.training.training_config import TrainingConfig
from corrector_src.model._cnn_mhd_corrector import CorrectorCNN
from corrector_src.model._cnn_mhd_corrector_options import (
    CNNMHDParams,
    CNNMHDconfig,
)

from corrector_src.training.loss import mse_loss

import equinox as eqx
import matplotlib.pyplot as plt
import optax
import time
from omegaconf import OmegaConf
import hydra
from hydra.utils import get_original_cwd
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def training_loop(cfg):
    print("Hydra run dir:", os.getcwd())
    num_cells_hr = cfg.data.hr_res
    downsampling_factor = cfg.data.downscaling_factor
    epochs = cfg.training.epochs
    n_look_behind = cfg.training.n_look_behind
    generate_data_on_fly = False
    
    # Training configuration
    training_config = TrainingConfig(
        compute_intermediate_losses=True,
        n_look_behind=n_look_behind,
        loss_function=mse_loss,
        loss_weights=None,
        use_relative_error=False,
    )

    model = CorrectorCNN(
        in_channels=8,
        hidden_channels=cfg.models.hidden_channels,
        key=jax.random.PRNGKey(60),
    )
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True, network_static=neural_net_static
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    snapshot_losses = []
    epoch_losses = []
    randomized_vars = [1, 1, 1]


    # Load ground truth data (TO BE CHANGED)
    def load_ground_truth(filepath="ground_truth.npy"):
        """
        Load the ground truth array from a saved numpy file.

        Args:
            filepath (str): Path to the saved numpy file. Default is 'ground_truth.npy'

        Returns:
            jnp.ndarray: The loaded ground truth array as a JAX array
        """
        try:
            ground_truth_np = np.load(filepath)
            ground_truth = jnp.array(ground_truth_np)

            print(f"Shape: {ground_truth.shape}")
            print(f"Data type: {ground_truth.dtype}")

            return ground_truth

        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return None
        except Exception as e:
            print(f"Error loading ground truth: {str(e)}")
            return None


    ground_truth = load_ground_truth(
        os.path.join(get_original_cwd(), "corrector_src", f"data/ground_truth_{cfg.data.num_checkpoints}.npy")
        
    )
    if ground_truth is None:
        (
            initial_state,
            config,
            params,
            helper_data,
            registered_variables,
            randomized_variables,
        ) = blast.randomized_initial_blast_state(num_cells_hr, randomized_vars)

        config = finalize_config(config, initial_state.shape)

        print("time integrating")
        final_states_hr = time_integration(
            initial_state, config, params, helper_data, registered_variables
        )

        ground_truth = final_states_hr.states
        ground_truth = downaverage_states(ground_truth, downsampling_factor)

        np.save(
            f"data/ground_truth_{cfg.data.num_checkpoints}.npy",
            np.array(ground_truth),
        )
    

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(float(cfg.training.learning_rate)),
    )
    opt_state = optimizer.init(neural_net_params)
    training_config._replace(ground_truth_snapshots=ground_truth)

    for i in range(epochs):
        if generate_data_on_fly:
            (
                initial_state,
                config,
                params,
                helper_data,
                registered_variables,
                _,
            ) = blast.randomized_initial_blast_state(num_cells_hr, randomized_vars)

            config = finalize_config(config, initial_state.shape)

            # print("time integrating")
            final_states_hr = time_integration(
                initial_state, config, params, helper_data, registered_variables
            )

            ground_truth = final_states_hr.states
            ground_truth = downaverage_states(ground_truth, downsampling_factor)
            training_config._replace(ground_truth_snapshots=ground_truth)

        time_config = time.time()
        initial_state, config, params, helper_data, registered_variables, _ = (
            blast.randomized_initial_blast_state(
                num_cells_hr // downsampling_factor, randomized_vars
            )
        )
        config = finalize_config(config, initial_state.shape)

        config = config._replace(cnn_mhd_corrector_config=cnn_mhd_corrector_config)
        params = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)
        time_config = time.time() - time_config
        time_train = time.time()
        losses, new_network_params, opt_state, final_full_sim_data = (
            time_integration_w_training(
                initial_state,
                config,
                params,
                helper_data,
                registered_variables,
                training_config,
                ground_truth,
                optimizer,
                opt_state,
                train_early_steps=False,
            )
        )
        time_train = time.time() - time_train
        snapshot_losses.append(losses.flatten())
        epoch_losses.append(np.mean(losses))
        cnn_mhd_corrector_params = cnn_mhd_corrector_params._replace(
            network_params=new_network_params
        )
        if jnp.isnan(final_full_sim_data.states).any():
            print("found nans in the final states running rollback")
            initial_state, config, params, helper_data, registered_variables, _ = (
                blast.randomized_initial_blast_state(
                    num_cells_hr // downsampling_factor, randomized_vars
                )
            )
            config = finalize_config(config, initial_state.shape)

            config = config._replace(cnn_mhd_corrector_config=cnn_mhd_corrector_config)
            params = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)
            final_states_rb = time_integration(
                initial_state, config, params, helper_data, registered_variables
            )
            if jnp.isnan(final_states_rb.states).any():
                print("nan found in rb stopping training")
                break
            else:
                print("nan not found in rb")
        print(
            f"epoch {i} time_config {time_config} time_train {time_train} loss {epoch_losses[-1]}"
        )

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
        n_look_behind = np.array(n_look_behind),
        epoch_losses=np.array(epoch_losses),
        snapshot_losses=np.array(snapshot_losses)   )

if __name__ == "__main__":
    training_loop()