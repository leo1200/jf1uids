#!/usr/bin/env bash
set -euo pipefail

# Create a temporary directory and work from there
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download and unpack micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# Install to ~/micromamba/bin
mkdir -p "$HOME/micromamba/bin"
mv bin/micromamba "$HOME/micromamba/bin/"
cd - > /dev/null
rm -rf "$TEMP_DIR"

# Update PATH for current shell and future shells
export PATH="$HOME/micromamba/bin:$PATH"
if ! grep -q 'micromamba/bin' "$HOME/.bashrc"; then
    echo 'export PATH=$HOME/micromamba/bin:$PATH' >> "$HOME/.bashrc"
fi

# Predefine MAMBA_ROOT_PREFIX to avoid unbound variable errors
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"

# Initialize micromamba shell integration safely
set +u
"$HOME/micromamba/bin/micromamba" shell init --shell bash --root-prefix="$MAMBA_ROOT_PREFIX"
set -u

# Add auto-activation of jf1uids to .bashrc if missing
if ! grep -q 'micromamba activate jf1uids' "$HOME/.bashrc"; then
    echo 'micromamba activate jf1uids' >> "$HOME/.bashrc"
fi

# Apply hook in current shell so activation works NOW
set +u
source ~/.bashrc
set -u

# Create the environment if not already present
if ! micromamba env list | grep -q '^jf1uids'; then
    micromamba create -y -n jf1uids python=3.10
fi

# activate the environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate jf1uids

# Install packages into the active environment
python -m pip install -U "jax[cuda12]"
python -m pip install autocvd
python -m pip install gpustat
python -m pip install matplotlib

echo
echo "✅ Setup complete!"
echo "➡️  Environment 'jf1uids' is active now."
echo "➡️  It will auto-activate in every new shell."
echo "➡️  For this session, if not sourced, run: source ~/.bashrc"
