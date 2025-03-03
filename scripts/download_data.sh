#!/bin/bash

cd /workspace/HyperDAS/axbench/axbench/data
python download-seed-sentences.py

bash /workspace/HyperDAS/experiments/axbench/download_gdrive.sh \
    "https://drive.google.com/file/d/1eNYXN0eAVmu2nuOqDNyiMI5IEq6faAcp/view\?usp\=sharing" /workspace/HyperDAS/axbench/axbench/concept16k

cd /workspace/HyperDAS