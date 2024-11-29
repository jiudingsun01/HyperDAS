#!/bin/bash
set -e

echo "🚀 Initializing HyperDAS container setup..."

# Add user to docker group if not already a member
if ! groups "$(id -un)" | grep -q "\bdocker\b"; then
    echo "👥 Adding user to docker group..."
    sudo adduser "$(id -un)" docker
    sg docker -c "$(readlink -f "$0") --continue"
    exit 0
fi

# Add a flag check to prevent infinite recursion
if [ "$1" != "--continue" ]; then
    if ! groups "$(id -un)" | grep -q "\bdocker\b"; then
        echo "❌ Error: Docker group permissions not applied. Please run the script again."
        exit 1
    fi
fi

# Cleanup function for containers and network
cleanup_containers() {
    echo "🧹 Cleaning up existing containers..."
    docker rm -f hyperdas ray-head 2>/dev/null || true
    docker network rm ray_network 2>/dev/null || true
}

# Source environment variables
ENV_FILE="$HOME/projects/HyperDAS/.env"
if [ -f "$ENV_FILE" ]; then
    echo "📝 Loading environment variables..."
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "❌ Error: .env file not found at $ENV_FILE"
    exit 1
fi

# Check required variables
if [ -z "$GIT_NAME" ] || [ -z "$GITHUB_TOKEN" ] || [ -z "$GIT_EMAIL" ]; then
    echo "❌ Error: Required environment variables not set"
    echo "ℹ️  Required: GIT_NAME, GIT_EMAIL, GITHUB_TOKEN"
    exit 1
fi

# Function to get remote image digest without pulling
get_remote_digest() {
    local image=$1
    echo "$GITHUB_TOKEN" | docker login ghcr.io -u $GIT_NAME --password-stdin >/dev/null 2>&1
    docker manifest inspect "$image" 2>/dev/null | grep -i '"digest"' | head -1 | tr -d ' ",' | cut -d':' -f2-3 || echo "none"
}

# Function to get local image digest
get_local_digest() {
    local image=$1
    local digest=$(docker image inspect "$image" --format='{{index .Id}}' 2>/dev/null | cut -d':' -f2 || echo "none")
    if [ "$digest" != "none" ]; then
        echo "sha256:$digest"
    else
        echo "none"
    fi
}

# Check if we need to update the image
echo "🔍 Checking for updates..."
LOCAL_DIGEST=$(get_local_digest "ghcr.io/$GIT_NAME/hyperdas:latest")
REMOTE_DIGEST=$(get_remote_digest "ghcr.io/$GIT_NAME/hyperdas:latest")

if [ "$LOCAL_DIGEST" != "$REMOTE_DIGEST" ]; then
    echo "🔄 New version detected, updating container..."
    cleanup_containers
    docker rmi ghcr.io/$GIT_NAME/hyperdas:latest 2>/dev/null || true
    docker pull "ghcr.io/$GIT_NAME/hyperdas:latest"
else
    echo "✨ Container is up to date"
    if docker ps -a | grep -q "hyperdas\|ray-head" && ! docker ps | grep -q hyperdas; then
        echo "🧹 Found stopped containers, removing them..."
        cleanup_containers
    elif docker ps | grep -q hyperdas; then
        echo "✅ Containers are already running"
        docker exec -it hyperdas /bin/bash
        exit 0
    fi
fi

# Login to GHCR
echo "🔑 Authenticating with GitHub Container Registry..."
echo "$GITHUB_TOKEN" | docker login ghcr.io -u $GIT_NAME --password-stdin

echo "🚀 Launching containers..."

# Create docker network if it doesn't exist
docker network create ray_network 2>/dev/null || true

echo "🐳 Starting containers..."
docker compose up -d

# Check if container is running
if docker ps | grep -q hyperdas; then
    echo "✅ Container started successfully!"
    echo "🎮 Available GPUs:"
    docker exec hyperdas nvidia-smi --list-gpus
    echo "🔌 Connecting to container..."
    docker exec -it hyperdas /bin/bash
else
    echo "❌ Error: Container failed to start"
    echo "📜 Container logs:"
    docker logs hyperdas
    exit 1
fi