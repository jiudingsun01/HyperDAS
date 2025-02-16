#!/bin/bash
set -e  # Exit on error

# Change to project directory if PROJECT_NAME is set
if [ ! -z "${PROJECT_NAME}" ]; then
    cd /workspace/${PROJECT_NAME}
fi

echo "🖥️  Container System Information:"
if [[ "$(uname -s)" == "Linux" ]]; then
    # Only run CUDA checks on Linux
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ CUDA is available"
        nvidia-smi
        # Get number of available GPUs
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo "📊 Found ${GPU_COUNT} GPU(s)"
        # Set CUDA_VISIBLE_DEVICES to all available GPUs (0,1,2,etc.)
        export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((GPU_COUNT-1)))
        echo "🎯 CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES}"
    else
        echo "⚠️  WARNING: CUDA is not available"
    fi
else
    echo "🍎 Running on macOS"
fi
echo "-----------------------------------"

# Print Python environment information
echo "🐍 Python Environment Information:"
python --version
pip list

# Print working directory information
echo "📂 Current working directory: $(pwd)"

# Move git config setup to beginning before any other operations
echo "🔧 Setting up git configuration..."

git config --global --replace-all user.email "${GIT_EMAIL}"
git config --global --replace-all user.name "${GIT_NAME}"
git config --global --replace-all safe.directory /workspace/${PROJECT_NAME}

echo "🚀 Container is ready!"
echo "-----------------------------------"

# Execute the command passed to docker run
echo "🔄 Executing command: $@"
exec "$@"