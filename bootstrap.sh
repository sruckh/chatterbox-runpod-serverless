#!/bin/bash
set -e

echo "=== ChatterBox Turbo Runpod Serverless Bootstrap ==="

# Create chatterbox directory structure on network volume
echo "Creating directory structure on network volume..."
mkdir -p /runpod-volume/chatterbox/{hf_home,hf_cache,models,output,audio_prompts}

# Set environment variables for HuggingFace cache
export HF_HOME="/runpod-volume/chatterbox/hf_home"
export HF_HUB_CACHE="/runpod-volume/chatterbox/hf_cache"

# Check if this is the first run
FIRST_RUN_FLAG="/runpod-volume/chatterbox/.first_run_complete"

if [ ! -f "$FIRST_RUN_FLAG" ]; then
    echo "=== First Run Detected - Setting up Environment ==="

    # Install PyTorch with CUDA 12.8 support
    echo "Installing PyTorch..."
    pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

    # Clone ChatterBox repository
    echo "Cloning ChatterBox repository..."
    cd /runpod-volume/chatterbox
    git clone https://github.com/resemble-ai/chatterbox.git

    # Modify pyproject.toml to remove conflicting dependencies
    echo "Modifying ChatterBox dependencies..."
    cd /runpod-volume/chatterbox/chatterbox

    # Create a temporary pyproject.toml without torch, torchaudio, and gradio
    python3 << 'EOF'
import toml

# Load original pyproject.toml
with open('pyproject.toml', 'r') as f:
    data = toml.load(f)

# Remove torch, torchaudio, and gradio from dependencies
if 'project' in data and 'dependencies' in data['project']:
    deps = data['project']['dependencies']
    # Remove packages that we'll install separately
    deps = [d for d in deps if not d.startswith(('torch', 'torchaudio', 'gradio'))]
    data['project']['dependencies'] = deps

# Save modified pyproject.toml
with open('pyproject.toml', 'w') as f:
    toml.dump(data, f)

print("Modified pyproject.toml successfully")
EOF

    # Install ChatterBox without conflicting dependencies
    echo "Installing ChatterBox..."
    cd /runpod-volume/chatterbox/chatterbox
    pip3 install -e .

    # Install additional requirements
    # We install them in Dockerfile but ensure here too just in case
    pip3 install runpod>=1.6.0 boto3>=1.26.0

    # Create first run flag
    touch "$FIRST_RUN_FLAG"
    echo "=== First Run Setup Complete ==="

else
    echo "=== Existing Installation Found - Skipping Setup ==="
fi

# Add ChatterBox to Python path
export PYTHONPATH="/runpod-volume/chatterbox/chatterbox:$PYTHONPATH"

# Start the handler
echo "Starting ChatterBox Turbo handler..."
exec python3 /workspace/chatterbox/handler.py
