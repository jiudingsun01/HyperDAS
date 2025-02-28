
#!/bin/bash
uv pip install --system --no-deps -e axbench
uv pip uninstall asyncio -y
uv pip install --system "numpy<2" --force-reinstall

git config --global --add safe.directory /workspace/HyperDAS/axbench
