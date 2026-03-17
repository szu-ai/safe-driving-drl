#!/usr/bin/env bash
set -e
python3 -u car.py   --mode eval   --host localhost   --port 2200   --target-town Town10HD_Opt   --target-weather mixed   --spawn-index -1   --eval-episodes 20   --checkpoint ./checkpoints/source_agent.pt   --out-dir ./culrt_carla_0915_aligned   --target-goal-index -1   --route-target-length 500   --npc-min 8   --npc-max 15   --debug 2>&1 | tee ./logs/eval_town10hd_500m_npc_log.txt
