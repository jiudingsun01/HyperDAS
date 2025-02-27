
#!/bin/bash
uv pip install --system --no-deps -e axbench
uv pip uninstall asyncio -y

git config --global --add safe.directory /workspace/HyperDAS/axbench
