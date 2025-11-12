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


from jf1uids import time_integration

from corrector_src.training.training_config import TrainingConfig, TrainingParams
from corrector_src.model.cnn_mhd_model import CorrectorCNN
from corrector_src.model._corrector_options import (
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
from corrector_src.data.dataset import dataset

import equinox as eqx
import matplotlib.pyplot as plt
import optax
import time
import hydra
from hydra.utils import get_original_cwd
from hydra.utils import instantiate
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def training_loop(cfg):
    print("Hydra run dir:", os.getcwd())
    epochs = cfg.training.epochs
    loss_timesteps = jnp.array(cfg.data.snapshot_timepoints)
    assert loss_timesteps.any() > cfg.data.t_end, (
        "found value greater than the end time in the snapshot timepoints"
    )

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

    # Create data
    rng_seed = 112
    gt_cfg_data = cfg.data
    gt_cfg_data.debug = False
    dataset_creator = dataset(gt_cfg_data.scenarios, gt_cfg_data)
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
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )

    @eqx.filter_jit
    def loss_fn(network_params_arrays):
        """Calculates the difference between the final state and the target."""
        results_low_res = time_integration(
            initial_state_lr,
            initial_config_lr,
            initial_params_lr._replace(
                cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                    network_params=network_params_arrays
                )
            ),
            initial_helper_data_lr,
            initial_registered_variables_lr,
        )
        # Calculate the L2 loss between the final state and the target state
        loss = jnp.mean((results_low_res.states - ground_truth) ** 2)
        return loss

    @eqx.filter_jit
    def train_step(network_params_arrays, opt_state):
        """Performs one step of gradient descent."""
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(network_params_arrays)
        updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
        network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
        return network_params_arrays, opt_state, loss_value

    print("Starting training with optax...")
    losses = []

    # This variable will hold the trained parameters and be updated in the loop
    trained_params = neural_net_params

    # Timing
    start_time = time.time()

    # # The main training loop
    pbar = tqdm(range(epochs))
    best_loss = float("inf")
    best_params = trained_params
    for step in pbar:
        trained_params, opt_state, loss = train_step(trained_params, opt_state)
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = trained_params
        pbar.set_description(f"Step {step + 1}/{epochs} | Loss: {loss:.2e}")

    # After training, use the best parameters found
    # trained_params = best_params

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    final_model = eqx.combine(best_params, neural_net_static)
    eqx.tree_serialise_leaves("cnn_model_leo.eqx", final_model)
    np.savez(
        "losses.npz",
        epoch_losses=np.array(losses),
    )


if __name__ == "__main__":
    training_loop()
