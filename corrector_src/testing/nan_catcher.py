# Test for the nan_flag implementation of time integration, this is useful to generate data on the fly and not put data nans accidentaly onto your data
from autocvd import autocvd

autocvd(num_gpus=1)

import os
import numpy as np
import jax.numpy as jnp
import jax


# corrector_src
from corrector_src.model.cnn_mhd_model import CorrectorCNN
from corrector_src.model.fno_hd_force_corrector import TurbulenceSGSForceCorrectorFNO
from corrector_src.model._corrector_options import (
    CorrectorConfig,
    CorrectorParams,
)

from corrector_src.data.dataset import dataset


# jf1uids
from jf1uids import time_integration

# other stuff
import equinox as eqx
import time
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

import matplotlib.pyplot as plt

main_experiment_folder = "../../experiments/turbulence_force_corrector/"
experiment_name = "2025-11-03_17-35-49"
experiment_path = os.path.join(main_experiment_folder, experiment_name)
config_path = os.path.join(experiment_path, ".hydra")


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def nan_catcher_test(cfg):
    rng_seed = 448505923  # 448505923 stable seed // 4158841521 nan seed
    num_snapshots = 30
    use_specific_snapshot_timepoints = True
    specific_snapshots = np.arange(0.0, cfg.data.t_end, cfg.data.t_end / 30).tolist()
    if cfg.data.t_end not in specific_snapshots:
        specific_snapshots.append(cfg.data.t_end)
    cfg.data.use_specific_snapshot_timepoints = use_specific_snapshot_timepoints
    cfg.data.num_snapshots = num_snapshots
    cfg.data.return_snapshots = True
    cfg.data.snapshot_timepoints = specific_snapshots
    create_fig = False
    with open_dict(cfg):
        cfg.data.differentiation_mode = 0

    dataset_turb = dataset([1], cfg.data)

    is_nan_data = True
    while is_nan_data:
        config_overrides = {
            "return_snapshots": True,
            "num_snapshots": 40,
            "use_specific_snapshot_timepoints": False,
            "active_nan_checker": True,
            "memory_analysis": True,
        }
        (sim_bundle_hr, sim_bundle_lr) = dataset_turb.hr_lr_initializator(
            resolution=cfg.data.hr_res,
            downscale=cfg.data.downscaling_factor,
            rng_seed=rng_seed,
            config_overrides=config_overrides,
        )
        print("starting hr integration")
        is_nan_data, states_hr = time_integration(**sim_bundle_hr.unpack_integrate())
        print(states_hr.states.shape, states_hr.states.nbytes)
        print("starting lr integration")
        is_nan_data, states_lr = time_integration(**sim_bundle_lr.unpack_integrate())
        print(states_lr.states.shape, states_lr.states.nbytes)
        if is_nan_data:
            print("nan found")
            continue

        print("data created")
    if create_fig:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(
            sim_bundle_hr.initial_state[2, :, :, sim_bundle_hr.config.num_cells // 2]
        )
        ax[1].imshow(
            sim_bundle_lr.initial_state[2, :, :, sim_bundle_lr.config.num_cells // 2]
        )
        plt.savefig(
            os.path.abspath(
                "/export/home/jalegria/Thesis/jf1uids/corrector/figures/initial_states.png"
            )
        )


if __name__ == "__main__":
    nan_catcher_test()
