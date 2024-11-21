#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/setup_lambda.sh <lambda_instance_ip>"
    exit 1
fi

LAMBDA_IP=$1
PROJECT_ROOT=$(git rev-parse --show-toplevel)

# Add this near the start of the script
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "📝 Loading local environment variables..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "🔄 Setting up Lambda Labs instance..."

# Create directories
ssh ubuntu@$LAMBDA_IP "mkdir -p ~/.config/hyperdas ~/projects/HyperDAS"

# Set correct permissions before syncing
echo "🔒 Setting correct SSH permissions..."
ssh ubuntu@$LAMBDA_IP "mkdir -p ~/.ssh && chmod 700 ~/.ssh"

# Sync credentials with correct permissions
echo "🔑 Syncing credentials..."
rsync -avz -e ssh \
    ~/.ssh/github* \
    ubuntu@$LAMBDA_IP:~/.ssh/ && \
ssh ubuntu@$LAMBDA_IP "chmod 600 ~/.ssh/github*"

# Sync environment file and append GITHUB_TOKEN if it exists locally
echo "📄 Syncing .env file..."
rsync -avz -e ssh \
    $PROJECT_ROOT/.env \
    ubuntu@$LAMBDA_IP:~/projects/HyperDAS/

# Sync code (excluding unnecessary files)
echo "📦 Syncing code..."
rsync -avz --progress -e ssh \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'assets' \
    --exclude '.env' \
    --exclude 'node_modules' \
    --exclude '.venv' \
    $PROJECT_ROOT/ \
    ubuntu@$LAMBDA_IP:~/projects/HyperDAS/

echo "🔧 Making scripts executable..."
ssh ubuntu@$LAMBDA_IP "chmod +x ~/projects/HyperDAS/scripts/*.sh"

echo "🚀 Starting container on Lambda instance..."
ssh -t ubuntu@$LAMBDA_IP "cd ~/projects/HyperDAS && ./scripts/lambdalabs.sh"

echo "✅ Setup complete!"