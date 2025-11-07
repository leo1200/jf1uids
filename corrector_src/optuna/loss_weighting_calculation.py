from autocvd import autocvd

autocvd(num_gpus=1)
import h5py
from corrector_src.loss.sgs_turb_loss import make_loss_function
from hydra import initialize, compose
from corrector_src.data.dataset import dataset
from omegaconf import OmegaConf
from jax import vmap
import jax.numpy as jnp
from jf1uids.time_stepping import time_integration
import os
from tqdm import tqdm


def main():
    with initialize(config_path="../../configs", version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=["data=turbulence_dataset", "training=turbulence_optuna"],
        )
    loss_weights_calculation(cfg)


def loss_weights_calculation(cfg):
    sim_bundle_creator = dataset(scenarios_to_use=cfg.data.scenarios, cfg_data=cfg.data)
    local_validation_bundle = sim_bundle_creator.sim_initializator(
        resolution=cfg.data.hr_res // cfg.data.downscaling_factor,
    )

    cfg_temp = OmegaConf.merge(
        cfg.training,
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

        for i, initial_state in enumerate(
            tqdm(initial_states, desc="Processing states")
        ):
            print(i)
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

    mean_components = {
        name: val / len(initial_states) for name, val in accumulated_components.items()
    }
    print("=" * 60)
    print(mean_components)
    print("=" * 60)


if __name__ == "__main__":
    main()
