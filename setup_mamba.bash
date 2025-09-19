# Create a temporary directory and work from there
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mkdir -p ~/micromamba/bin
mv bin/micromamba ~/micromamba/bin/
cd - > /dev/null
rm -rf "$TEMP_DIR"

# Add to PATH (bash/zsh)
echo 'export PATH=$HOME/micromamba/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# one-time setup for bash
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba

# restart shell
exec bash

# create the environment for jf1uids
micromamba create -n jf1uids python=3.10
micromamba activate jf1uids

# install jax with cuda support
python -m pip install -U "jax[cuda12]"

# install autocvd
python -m pip install autocvd

# install gpustat
python -m pip install gpustat

# install matplotlib
python -m pip install matplotlib