#!/bin/bash

cd /workspace/HyperDAS/assets/data/axbench
python /workspace/HyperDAS/axbench/axbench/data/download-seed-sentences.py
bash /workspace/HyperDAS/axbench/axbench/data/download-alpaca.sh

bash /workspace/HyperDAS/experiments/axbench/download_gdrive.sh \
    "https://drive.google.com/file/d/1eNYXN0eAVmu2nuOqDNyiMI5IEq6faAcp/view\?usp\=sharing" /workspace/HyperDAS/axbench/axbench/concept16k

cd /workspace/HyperDAS
