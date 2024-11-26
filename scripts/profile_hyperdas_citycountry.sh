#!/bin/bash

export WANDB_RUN_GROUP="profiler_runs"

python train.py \
  --inference_modes default \
  --num_decoders 8 \
  --intervention_layer 15 \
  --batch_size 32 \
  --lr 3e-5 \
  --test_path ./experiments/RAVEL/data/city_country_test \
  --train_path ./experiments/RAVEL/data/city_country_train \
  --save_dir ./assets/checkpoints \
  --wandb_run_name fix_hyperdas_profile_city_country \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --log_wandb