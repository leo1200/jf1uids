from autocvd import autocvd

autocvd(num_gpus=1)


from corrector_src.data.dataset import dataset
import os
os.environ["JAX_LOG_COMPILES"] = "1"
import h5py
from hydra import initialize, compose
import gc
from jax import clear_caches, log_compiles

def main():
    with initialize(config_path="../../configs", version_base="1.2"):
        cfg = compose(config_name="config", overrides=["data=turbulence_dataset"])

    dataset_creation(cfg)


def dataset_creation(cfg):
    save_path = os.path.abspath("/export/data/jalegria/jf1uids_turbulence_sol")
    os.makedirs(save_path, exist_ok=True)

    h5_path = os.path.join(save_path, "validation_data_turbulence.h5")
    h5f = h5py.File(h5_path, "w")

    ground_truth_shape = (
        cfg.data.n_simulations,
        cfg.data.num_snapshots,
        5,  # density + pressure + 3 velocities
        cfg.data.hr_res // cfg.data.downscaling_factor,
        cfg.data.hr_res // cfg.data.downscaling_factor,
        cfg.data.hr_res // cfg.data.downscaling_factor,
    )
    initial_state_shape = (
        cfg.data.n_simulations,
        5,  # density + pressure + 3 velocities
        cfg.data.hr_res // cfg.data.downscaling_factor,
        cfg.data.hr_res // cfg.data.downscaling_factor,
        cfg.data.hr_res // cfg.data.downscaling_factor,
    )

    gt_dataset = h5f.create_dataset(
        "gt_states", shape=ground_truth_shape, dtype="float32"
    )
    initial_state_dataset = h5f.create_dataset(
        "initial_state", shape=initial_state_shape, dtype="float32"
    )
    energy_dataset = h5f.create_dataset(
        "first_snapshot_energy",
        shape=(cfg.data.n_simulations),
        dtype="float32",
    )
    mass_dataset = h5f.create_dataset(
        "first_snapshot_mass",
        shape=(cfg.data.n_simulations),
        dtype="float32",
    )

    seed_dataset = h5f.create_dataset(
        "seed",
        shape=(cfg.data.n_simulations),
        dtype="int",
    )

    dataset_creator = dataset(scenarios_to_use=cfg.data.scenarios, cfg_data=cfg.data)
    for i in range(cfg.data.n_simulations):
        print(i)
        with log_compiles():
            (
                hr_downscaled_states,
                initial_state_lr,
                total_energy,
                total_mass,
                seed,
            ) = dataset_creator.dataset_validation_initializator()

        gt_dataset[i] = hr_downscaled_states
        initial_state_dataset[i] = initial_state_lr
        energy_dataset[i] = total_energy
        mass_dataset[i] = total_mass
        seed_dataset[i] = seed
        del hr_downscaled_states, initial_state_lr, total_energy, total_mass, seed 
        gc.collect()
        if i % 20 == 0:
            h5f.flush()
            clear_caches()
    h5f.close()


if __name__ == "__main__":
    main()
