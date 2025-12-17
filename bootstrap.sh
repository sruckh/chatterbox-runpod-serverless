#!/bin/bash
set -e

echo "=== ChatterBox Turbo Runpod Serverless Bootstrap ==="

# Create chatterbox directory structure on network volume
echo "Creating directory structure on network volume..."
mkdir -p /runpod-volume/chatterbox/{hf_home,hf_cache,models,output,audio_prompts}

# Set environment variables for HuggingFace cache
export HF_HOME="/runpod-volume/chatterbox/hf_home"
export HF_HUB_CACHE="/runpod-volume/chatterbox/hf_cache"

# Virtual Environment Path on Network Volume
VENV_PATH="/runpod-volume/chatterbox/venv"

# Check if this is the first run
FIRST_RUN_FLAG="/runpod-volume/chatterbox/.first_run_complete"

if [ ! -f "$FIRST_RUN_FLAG" ]; then
    echo "=== First Run Detected - Setting up Environment ==="

    # Create Virtual Environment
    echo "Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    
    # Activate Virtual Environment
    source "$VENV_PATH/bin/activate"

    # Install PyTorch with CUDA 12.8 support
    echo "Installing PyTorch..."
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

    # Clone ChatterBox repository
    echo "Cloning ChatterBox repository..."
    cd /runpod-volume/chatterbox
    # Remove if exists to ensure clean slate on "first run" logic (e.g. if failed previously)
    rm -rf chatterbox
    git clone https://github.com/resemble-ai/chatterbox.git

    # Modify pyproject.toml to remove conflicting dependencies
    echo "Modifying ChatterBox dependencies..."
    cd /runpod-volume/chatterbox/chatterbox

    # Install toml for the modification script
    pip install toml

    # Create a temporary pyproject.toml without torch, torchaudio, and gradio
    python << 'EOF'
import toml

# Load original pyproject.toml
with open('pyproject.toml', 'r') as f:
    data = toml.load(f)

# Remove torch, torchaudio, and gradio from dependencies
if 'project' in data and 'dependencies' in data['project']:
    deps = data['project']['dependencies']
    new_deps = []
    for d in deps:
        # Remove packages that we'll install separately
        if d.startswith(('torch', 'torchaudio', 'gradio')):
            continue
        # Update numpy to be compatible with Python 3.12
        if d.startswith('numpy'):
            new_deps.append('numpy>=1.26.0')
        else:
            new_deps.append(d)
    data['project']['dependencies'] = new_deps

# Save modified pyproject.toml
with open('pyproject.toml', 'w') as f:
    toml.dump(data, f)

print("Modified pyproject.toml successfully")
EOF

    # Install ChatterBox without conflicting dependencies
    echo "Installing ChatterBox..."
    cd /runpod-volume/chatterbox/chatterbox
    pip install -e .

    # Install additional requirements
    # We install them in Dockerfile but ensure here too just in case
    pip install runpod>=1.6.0 boto3>=1.26.0

    # Create first run flag
    touch "$FIRST_RUN_FLAG"
    echo "=== First Run Setup Complete ==="

else
    echo "=== Existing Installation Found - Skipping Setup ==="
    # Activate Virtual Environment
    source "$VENV_PATH/bin/activate"
fi

# Start the handler
echo "Starting ChatterBox Turbo handler..."
exec python /workspace/chatterbox/handler.py
