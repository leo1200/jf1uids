import os
import h5py
import yaml

config_file = yaml.safe_load(
    open("./home/jalegria/Thesis/jf1uids/corrector_src/config.yaml")
)
save_path = "./data/jalegria/corrector/"
os.makedirs(save_path, exist_ok=True)

h5_path = os.path.join(save_path, "simulations.h5")
h5f = h5py.File(h5_path, "w")
