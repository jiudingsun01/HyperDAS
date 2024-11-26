#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/setup_conda_remote.sh <remote_ip> [environment_name]"
    exit 1
fi

REMOTE_IP=$1
ENV_NAME=${2:-$(conda info --envs | grep '*' | awk '{print $1}')}  # Use provided name or get current env
PROJECT_ROOT=$(git rev-parse --show-toplevel)
PYTHON_VERSION=$(python -V | cut -d' ' -f2)  # Get current Python version

# ... existing env loading code ...

echo "🔄 Setting up conda environment on remote machine..."

# Install Miniconda if not already installed
echo "📦 Ensuring Miniconda is installed..."
ssh ubuntu@$REMOTE_IP "if [ ! -f ~/miniconda3/bin/conda ]; then \
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    echo 'export PATH=~/miniconda3/bin:\$PATH' >> ~/.bashrc && \
    ~/miniconda3/bin/conda init bash && \
    source ~/.bashrc; \
fi"

# Create conda environment with same Python version only if it doesn't exist
echo "🐍 Setting up conda environment: $ENV_NAME..."
ssh ubuntu@$REMOTE_IP "export PATH=~/miniconda3/bin:\$PATH && \
    if ! ~/miniconda3/bin/conda env list | grep -q '^$ENV_NAME '; then \
        if [ -f ~/projects/$ENV_NAME/environment.yml ]; then \
            ~/miniconda3/bin/conda env create -f ~/projects/$ENV_NAME/environment.yml -n $ENV_NAME; \
        else \
            ~/miniconda3/bin/conda create -y -n $ENV_NAME python=$PYTHON_VERSION; \
        fi \
    else \
        echo 'Environment already exists, skipping creation...'; \
    fi && \
    ~/miniconda3/bin/conda run -n $ENV_NAME pip freeze > /tmp/requirements.txt"

# Get the project directory name from the root path
PROJECT_DIR=$(basename $PROJECT_ROOT)

# Create projects directory with project name (not environment name)
echo "📁 Creating project directory..."
ssh ubuntu@$REMOTE_IP "mkdir -p ~/projects/$PROJECT_DIR"

# Sync project code using PROJECT_DIR instead of ENV_NAME
echo "📦 Syncing code..."
rsync -avz --progress -e ssh \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'node_modules' \
    --exclude '.venv' \
    --exclude '*.egg-info' \
    $PROJECT_ROOT/ \
    ubuntu@$REMOTE_IP:~/projects/$PROJECT_DIR/

# Install dependencies using PROJECT_DIR
echo "📚 Installing dependencies..."
ssh ubuntu@$REMOTE_IP "export PATH=~/miniconda3/bin:\$PATH && \
    cd ~/projects/$PROJECT_DIR && \
    if [ ! -f environment.yml ] && [ -f requirements.txt ]; then \
        ~/miniconda3/bin/conda run -n $ENV_NAME pip install -r requirements.txt; \
    fi && \
    if [ -f setup.py ]; then \
        ~/miniconda3/bin/conda run -n $ENV_NAME pip install -e .; \
    fi"

# Initialize conda in .bashrc if not already done
echo "🔧 Ensuring conda is initialized..."
ssh ubuntu@$REMOTE_IP "~/miniconda3/bin/conda init bash && source ~/.bashrc"

echo "✅ Setup complete! Connecting to remote environment..."
# Create a temporary script that initializes conda and activates the environment
ssh ubuntu@$REMOTE_IP "cat > /tmp/conda_init.sh << 'EOF'
#!/bin/bash
eval \"\$(~/miniconda3/bin/conda shell.bash hook)\"
conda activate $ENV_NAME
exec bash
EOF
chmod +x /tmp/conda_init.sh"

# Connect and run the initialization script
ssh -t ubuntu@$REMOTE_IP "bash /tmp/conda_init.sh"