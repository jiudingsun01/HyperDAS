#!/bin/bash

export WANDB_RUN_GROUP="profiler_runs"

python train_baseline.py \
  --batch_size 32 \
  --intervention_layer 15 \
  --save_dir ./assets/checkpoints \
  --test_path ./experiments/RAVEL/data/city_country_test \
  --train_path ./experiments/RAVEL/data/city_country_train \
  --wandb_run_name mdas_profile_city_country \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --log_wandb