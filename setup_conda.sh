#!/bin/bash

# Exit script on error
set -e

# Variables
CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
CONDA_URL="https://repo.anaconda.com/miniconda/$CONDA_INSTALLER"
INSTALL_DIR="$HOME/miniconda3"
ENV_NAME="myenv"
PYTHON_VERSION="3.9"
PACKAGES="numpy pandas matplotlib"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Starting Miniconda installation and Conda environment setup..."

# Step 1: Download Miniconda installer
if [ ! -f "$CONDA_INSTALLER" ]; then
    echo "Downloading Miniconda installer..."
    curl -O "$CONDA_URL"
else
    echo "Miniconda installer already downloaded."
fi

# Step 2: Install Miniconda
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Installing Miniconda..."
    bash "$CONDA_INSTALLER" -b -p "$INSTALL_DIR"
else
    echo "Miniconda already installed at $INSTALL_DIR."
fi

# Step 3: Initialize Conda
echo "Initializing Conda..."
eval "$($INSTALL_DIR/bin/conda shell.bash hook)"

# Step 4: Update Conda to the latest version
echo "Updating Conda..."
conda update -n base -c defaults conda -y

# Step 5: Create a new Conda environment
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION and packages: $PACKAGES..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" $PACKAGES -y
fi

# Step 6: Activate the environment and verify
echo "Activating Conda environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "Environment setup complete. Here is the environment info:"
conda list

# Optional: Clean up the installer
echo "Cleaning up..."
rm -f "$CONDA_INSTALLER"

echo "All done! To activate your environment in the future, use: conda activate $ENV_NAME"