#!/bin/bash

# Simple script to check for conda and install miniconda3 if not found

set -e  # Exit on any error

# Configuration
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh"
INSTALL_DIR="$HOME/scratch/miniconda3"

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        echo "✓ Conda found at: $(which conda)"
        conda --version
        return 0
    else
        echo "✗ Conda not found"
        return 1
    fi
}

# Function to install miniconda3
install_miniconda() {
    echo "Installing Miniconda3..."

    # Create temporary directory for download
    temp_dir="conda_tmp"
    mkdir -p "$temp_dir"
    installer_path="$temp_dir/Miniconda3-latest-Linux-ppc64le.sh"

    # Download miniconda installer
    echo "Downloading Miniconda3 installer..."
    if command -v wget &> /dev/null; then
        wget -O "$installer_path" "$MINICONDA_URL"
    elif command -v curl &> /dev/null; then
        curl -o "$installer_path" "$MINICONDA_URL"
    else
        echo "Error: Neither wget nor curl found. Please install one of them."
        exit 1
    fi

    # Make installer executable
    chmod +x "$installer_path"

    # Run installer
    echo "Running Miniconda3 installer..."
    bash "$installer_path" -b -p "$INSTALL_DIR"

    # Clean up
    rm -rf "$temp_dir"

    echo "✓ Miniconda3 installed successfully"
    # echo "Note: Please restart your terminal or run 'source ~/.bashrc' to use conda in new sessions"

    # Manual initialization
    echo "Initializing conda..."
    source $INSTALL_DIR/bin/activate
    conda init --all
}

# Main execution
echo "=== Conda Installation Check ==="

if check_conda; then
    echo "Conda is already installed. Nothing to do."
else
    echo "Installing Miniconda3..."
    install_miniconda

    # Verify installation
    echo "Verifying installation..."
    if check_conda; then
        echo "✓ Miniconda3 installation successful"
    else
        echo "✗ Miniconda3 installation failed"
        exit 1
    fi
fi

echo "=== Done ==="