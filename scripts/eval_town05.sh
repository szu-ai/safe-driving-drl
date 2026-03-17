#!/usr/bin/env bash
set -e
python3 -u car.py   --mode eval   --host localhost   --port 2200   --target-town Town05   --target-weather mixed   --spawn-index 0   --eval-episodes 20   --checkpoint ./checkpoints/source_agent.pt   --out-dir ./culrt_carla_0915_aligned   --target-goal-index -1   --debug 2>&1 | tee ./logs/eval_town05_log.txt
