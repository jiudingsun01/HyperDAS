#!/bin/bash
uv pip install --system hatchling editables
uv pip install --system -e /workspace/HyperDAS/axbench --no-build-isolation
uv pip uninstall --system asyncio
uv pip install --system "numpy<2" --force-reinstall 

git config --global --add safe.directory /workspace/HyperDAS/axbench
