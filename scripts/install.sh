
#!/bin/bash
uv pip install --system -e axbench --no-build-isolation
pip uninstall asyncio -y
uv pip install --system "numpy<2" --force-reinstall 

git config --global --add safe.directory /workspace/HyperDAS/axbench
bash experiments/axbench/download_gdrive.sh https://drive.google.com/file/d/1eNYXN0eAVmu2nuOqDNyiM
I5IEq6faAcp/view\?usp\=sharing axbench/axbench/concept16k