#!/usr/bin/env bash
set -e
python3 -u car.py   --mode train   --host localhost   --port 2200   --tm-port 8001   --train-town Town10HD_Opt   --spawn-index -1   --train-goal-index -1   --train-steps 500000   --train-npc-min 8   --train-npc-max 20   --source-weather night_rain_fog   --out-dir ./culrt_carla_0915_aligned   --maml-warmup-batches 10   --debug 2>&1 | tee ./logs/train_paper_log.txt
