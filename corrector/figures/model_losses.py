import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from autocvd import autocvd

autocvd(num_gpus=1)

import jax.numpy as jnp
import jax
import equinox as eqx

from corrector_src.model.cnn_mhd_model import CorrectorCNN
from corrector_src.model._cnn_mhd_corrector_options import (
    CNNMHDParams,
    CNNMHDconfig,
)
from corrector_src.utils.downaverage import downaverage_states
import hydra
from corrector_src.data.load_sim import (
    prepare_initial_state,
)
from jf1uids import time_integration

import matplotlib.pyplot as plt
import numpy as np

src_folder = "/export/home/jalegria/Thesis/jf1uids"
experiment_folder = "experiments/experiment_1/2025-10-14_10-25-28_10"
abs_experiment_folder = os.path.join(src_folder, experiment_folder)
experiment_name = "_losses_leo.png"


@hydra.main(
    version_base=None,
    config_path=os.path.join("../..", experiment_folder, ".hydra"),
    config_name="config",
)
def plot_losses(cfg):
    print(cfg)
    # Create a test model
    # /export/home/jalegria/Thesis/jf1uids/experiments/experiment_1/2025-10-13_12-09-18_10 first try with gradient accumulation
    model_name = "cnn_model_leo.eqx"
    key = jax.random.PRNGKey(42)
    model = CorrectorCNN(in_channels=8, hidden_channels=16, key=key)
    model = eqx.tree_deserialise_leaves(
        os.path.join(abs_experiment_folder, model_name),
        model,
    )
    # # Test input
    # test_input = jnp.ones((8, 32, 32, 32))

    # # Test forward pass
    # output = model(test_input)
    # print(f"Output shape: {output.shape}")
    # print(f"Output norm: {jnp.linalg.norm(output)}")
    # del output, test_input

    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True, network_static=neural_net_static
    )
    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    cfg_data = cfg.data
    cfg_data.use_specific_snapshot_timepoints = True
    cfg_data.snapshot_timepoints = np.linspace(0.0, cfg_data.t_end, 100).tolist()
    rng_seed = 112
    (
        lr_initial_state,
        lr_config_corrected,
        lr_params_corrected,
        lr_helper_data_corrected,
        lr_registered_variables_corrected,
    ) = prepare_initial_state(
        cfg_data=cfg_data,
        rng_seed=rng_seed,
        cnn_mhd_corrector_config=cnn_mhd_corrector_config,
        cnn_mhd_corrector_params=cnn_mhd_corrector_params,
        downscale=True,
    )
    (
        lr_initial_state,
        lr_config_not_corrected,
        lr_params_not_corrected,
        lr_helper_data_not_corrected,
        lr_registered_variables_not_corrected,
    ) = prepare_initial_state(
        cfg_data=cfg_data,
        rng_seed=rng_seed,
        cnn_mhd_corrector_config=None,
        cnn_mhd_corrector_params=None,
        downscale=True,
    )

    hr_initial_state, hr_config, hr_params, hr_helper_data, hr_registered_variables = (
        prepare_initial_state(
            cfg_data=cfg_data,
            rng_seed=rng_seed,
            cnn_mhd_corrector_config=None,
            cnn_mhd_corrector_params=None,
            downscale=False,
        )
    )
    print(jnp.shape(lr_initial_state), jnp.shape(hr_initial_state))

    # lr_config_corrected = lr_config_corrected._replace(progress_bar=True)
    # lr_config_not_corrected = lr_config_not_corrected._replace(progress_bar=True)
    # hr_config = hr_config._replace(progress_bar=True)
    lr_corrected_snapshot_data = time_integration(
        lr_initial_state,
        lr_config_corrected,
        lr_params_corrected,
        lr_helper_data_corrected,
        lr_registered_variables_corrected,
    )
    lr_not_corrected_snapshot_data = time_integration(
        lr_initial_state,
        lr_config_not_corrected,
        lr_params_not_corrected,
        lr_helper_data_not_corrected,
        lr_registered_variables_not_corrected,
    )

    hr_snapshot_data = time_integration(
        hr_initial_state, hr_config, hr_params, hr_helper_data, hr_registered_variables
    )

    hr_states = downaverage_states(hr_snapshot_data.states, cfg.data.downscaling_factor)
    lr_corrected_states = lr_corrected_snapshot_data.states
    lr_not_corrected_states = lr_not_corrected_snapshot_data.states

    # MSE per snapshot, same as training
    loss_lr_hr_corrected = jnp.mean(
        (lr_corrected_states - hr_states) ** 2, axis=(1, 2, 3, 4)
    )

    loss_lr_lr = jnp.mean(
        (lr_corrected_states - lr_not_corrected_states) ** 2, axis=(1, 2, 3, 4)
    )

    loss_lr_hr_not_corrected = jnp.mean(
        (lr_not_corrected_states - hr_states) ** 2, axis=(1, 2, 3, 4)
    )

    losses_training = np.load(
        os.path.join(abs_experiment_folder, "losses.npz"), allow_pickle=True
    )

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))  # adjust figsize as needed

    # --- First subplot: snapshot losses ---
    used_snapshot_times_training = cfg.data.snapshot_timepoints
    snapshot_times = np.array(
        cfg_data.snapshot_timepoints
    )  # convert ListConfig to np.array
    trained_times_index = []

    for i in range(len(used_snapshot_times_training)):
        axs[0].plot(
            losses_training["snapshot_losses"][i::2],
            label=str(used_snapshot_times_training[i]),
        )
        trained_times_index.append(
            np.argmin(np.abs(snapshot_times - used_snapshot_times_training[i]))
        )

    axs[0].set_title("Snapshot Losses")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # --- Second subplot: comparison losses ---
    axs[1].plot(
        cfg_data.snapshot_timepoints, loss_lr_hr_corrected, label="lr corrected - hr"
    )
    axs[1].plot(cfg_data.snapshot_timepoints, loss_lr_lr, label="lr corrected - lr")
    axs[1].plot(cfg_data.snapshot_timepoints, loss_lr_hr_not_corrected, label="lr - hr")

    for time in used_snapshot_times_training:
        axs[1].axvline(time, color="gray", linestyle="--")

    axs[1].set_title("Comparison of Losses over Snapshot Times")
    axs[1].set_xlabel("Snapshot Time")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.abspath(
            "/export/home/jalegria/Thesis/jf1uids/corrector/figures"
            + "/losses_blast"
            + experiment_name
        )
    )


if __name__ == "__main__":
    plot_losses()
