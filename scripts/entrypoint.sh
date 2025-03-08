#!/bin/bash
set -e # Exit on error

# Create NVIDIA device symlinks if nvidia-ctk is available
if command -v nvidia-ctk &>/dev/null; then
    echo "🔧 Creating NVIDIA device symlinks..."
    nvidia-ctk system create-dev-char-symlinks --create-all || true
fi

# Change to project directory if PROJECT_NAME is set
if [ ! -z "${PROJECT_NAME}" ]; then
    cd /workspace/${PROJECT_NAME}
fi

echo "🖥️  Container System Information:"
if [[ "$(uname -s)" == "Linux" ]]; then
    # Only run CUDA checks on Linux
    if command -v nvidia-smi &>/dev/null; then
        echo "✅ CUDA is available"
        nvidia-smi
        # Get number of available GPUs
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo "📊 Found ${GPU_COUNT} GPU(s)"
        # Set CUDA_VISIBLE_DEVICES to all available GPUs (0,1,2,etc.)
        export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((GPU_COUNT - 1)))
        echo "🎯 CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES}"
    else
        echo "⚠️  WARNING: CUDA is not available"
        # Check for common issues
        echo "Checking for common GPU access issues..."
        if [ ! -e /dev/nvidia0 ]; then
            echo "❌ /dev/nvidia0 device not found"
        fi
        if [ ! -e /dev/nvidiactl ]; then
            echo "❌ /dev/nvidiactl device not found"
        fi
        if [ ! -e /dev/nvidia-uvm ]; then
            echo "❌ /dev/nvidia-uvm device not found"
        fi
    fi
else
    echo "🍎 Running on macOS"
fi
echo "-----------------------------------"

# Print Python environment information
echo "🐍 Python Environment Information:"
python --version

# Print working directory information
echo "📂 Current working directory: $(pwd)"

# Move git config setup to beginning before any other operations
echo "🔧 Setting up git configuration..."
git config --global --replace-all user.email "${GIT_EMAIL}"
git config --global --replace-all user.name "${GIT_NAME}"
git config --global --replace-all safe.directory /workspace/${PROJECT_NAME}

# Save SSH agent environment variables to a file that can be sourced
echo "🔑 Setting up SSH agent environment..."
echo "export SSH_AUTH_SOCK=$SSH_AUTH_SOCK" > /etc/profile.d/ssh-agent.sh
echo "export SSH_AGENT_PID=$SSH_AGENT_PID" >> /etc/profile.d/ssh-agent.sh
chmod +x /etc/profile.d/ssh-agent.sh

# Also add to .bashrc and .zshrc for interactive shells
echo "source /etc/profile.d/ssh-agent.sh" >> /root/.bashrc
if [ -f /root/.zshrc ]; then
    echo "source /etc/profile.d/ssh-agent.sh" >> /root/.zshrc
fi

echo "🔑 Loading SSH keys..."
# Load SSH keys (ssh-agent is already running from the ENTRYPOINT)
for key in ~/.ssh/*; do
    # Skip public keys, authorized_keys, known_hosts, and config files
    if [[ -f "$key" && $key != *.pub && $key != */authorized_keys && $key != */known_hosts && $key != */config ]]; then
        echo "Adding key: $key"
        ssh-add "$key" 2>/dev/null || echo "Could not add key: $key"
    fi
done

# Test GitHub SSH connection but don't exit if it fails
echo "🔍 Testing GitHub SSH connection..."
if ssh -T -o "StrictHostKeyChecking=no" git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "✅ GitHub SSH connection successful"
else
    echo "⚠️ GitHub SSH connection failed (continuing anyway)"
fi

echo "🚀 Container is ready!"
echo "-----------------------------------"

# Execute the command passed to docker run
echo "🔄 Executing command: $@"
exec "$@"
