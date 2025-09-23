#!/usr/bin/env bash
set -euo pipefail

# Install micromamba to ~/micromamba/bin
mkdir -p "$HOME/micromamba/bin"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C "$HOME/micromamba/bin" bin/micromamba --strip-components=1

# Ensure micromamba is on PATH
if ! grep -q 'micromamba/bin' "$HOME/.bashrc"; then
    echo 'export PATH=$HOME/micromamba/bin:$PATH' >> "$HOME/.bashrc"
fi
export PATH="$HOME/micromamba/bin:$PATH"

# Root prefix
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"

# Ensure micromamba shell hook is in .bashrc
if ! grep -q 'micromamba shell hook' "$HOME/.bashrc"; then
    echo 'eval "$($HOME/micromamba/bin/micromamba shell hook --shell bash)"' >> "$HOME/.bashrc"
fi

# Ensure auto-activation of jf1uids is in .bashrc (after the hook)
if ! grep -q 'micromamba activate jf1uids' "$HOME/.bashrc"; then
    echo 'micromamba activate jf1uids' >> "$HOME/.bashrc"
fi

# Enable shell functions in this script run
eval "$($HOME/micromamba/bin/micromamba shell hook --shell bash)"

# Create env if missing
if ! micromamba env list | grep -q '^jf1uids'; then
    micromamba create -y -n jf1uids python=3.10
fi

# Activate now
micromamba activate jf1uids

# Install Python packages
python -m pip install -U --no-cache-dir "jax[cuda12]" autocvd gpustat matplotlib 

echo "✅ Setup complete. Environment 'jf1uids' active."
echo "➡️  It will auto-activate in every new shell."
