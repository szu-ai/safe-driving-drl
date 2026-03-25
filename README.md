# Reliable Policy Transfer for Safety-Aware End-to-End Driving with Deep Reinforcement Learning

**CVPR 2026 · Submission 38759**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange)](https://carla.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Uddin Md. Borhan, Arif Raza, Zhiliang Lin, Lu Wang, Jianqiang Li, Jie Chen**  
> College of Computer Science and Software Engineering, Shenzhen University  
> Corresponding author: chenjie@szu.edu.cn

---

## Overview

This repository contains the official implementation of a unified Deep Reinforcement Learning (DRL) framework for **safety-aware end-to-end autonomous driving with reliable policy transfer** in **CARLA 0.9.15**.

The framework addresses four tightly coupled challenges in closed-loop autonomous driving:

1. **Ego-centric relational state** — An uncertainty-weighted attention graph captures causal interactions between the ego vehicle and nearby agents, making safety-critical influences explicit to the policy.
2. **Differentiable multi-objective reward shaping** — Dense reward terms jointly optimize safety, progress, comfort, and uncertainty-aware behavior, avoiding unstable sparse event-only penalties.
3. **Uncertainty-gated exploration** — Aleatoric and epistemic uncertainty are combined into a calibrated confidence signal that adaptively modulates policy entropy for risk-aware exploration.
4. **Causal-semantic policy transfer** — Transfer learning aligns action distributions, relational attention, and uncertainty statistics across source and target domains, together with meta-initialization for fast adaptation.

---

## Unified System Model

<p align="center">
  <img src="./diagrams/unified_framework.png" width="90%" alt="Unified Framework"/>
</p>

This unified framework integrates ego-centric relational state construction, dense multi-objective reward shaping, uncertainty-gated SAC exploration, and causal-semantic transfer learning.

---

## Closed-Loop Route Coverage

<p align="center">
  <img src="./close_loop.png" width="55%" alt="Closed-Loop Route Coverage Map"/>
</p>

Closed-loop trajectory coverage on Town10HD (source domain). Blue trajectories indicate successful runs; red indicates unsuccessful. Dashed lines show planned routes.

---

## Key Results

Closed-loop evaluation in CARLA 0.9.15 across Town10HD (source), Town02, and Town05 (targets) under adverse weather.

| Map / Setting | SR (%) | DS | IS | Coll./km | Off/km | TO/km |
|---|---:|---:|---:|---:|---:|---:|
| Town10HD (source training) | 91.2 | 94.1 | 1.00 | 0.000 | 0.000 | 0.000 |
| Town05 (zero-shot transfer) | 100.0 | 94.1 | 1.00 | 0.000 | 0.000 | 0.000 |
| Town02 — Policy Learning | — | 188.6 | 0.88 | 0.007 | 0.005 | 0.003 |
| Town02 — Source Domain | — | 205.7 | 0.92 | 0.006 | 0.004 | 0.002 |
| Town02 — **Target (full transfer)** | — | **214.3** | **0.94** | **0.005** | **0.003** | **0.001** |

**SR** = Success Rate, **DS** = Driving Score, **IS** = Infraction Score.  
Town05 zero-shot: mean CTE = 0.183 m, mean heading error = 0.021 rad.

---

## Architecture

```text
Perception (CARLA sensor stack)
        │
        ▼
Ego-Centric Relational Graph
  edges: [Δp, Δv, class, κ, σ²_ale]
  attention: α_i = softmax(−‖Δp_i‖² / (σ²_i + ε))
  state: s_t = [z_t; v_ego; a^{t−1}; d_goal; φ_lane; σ²_ale]
        │
        ▼
SAC Actor-Critic
  uncertainty: σ²_dec = σ²_ale + σ²_epi
  entropy gate: β(σ̄) = β₀(1 − σ̄)
        │
        ├─── Dense Reward
        │    r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u
        │
        └─── Transfer (source→target)
             L_trans = L_KL + λ_α MMD(α_s, α_t) + λ_u ‖u_s − u_t‖²
             + MAML initialization
```

---

## Method Details

### 1. Ego-centric relational state
Each scene entity contributes an edge representation containing relative kinematics, semantic type, lane geometry, and aleatoric uncertainty. Uncertainty-weighted attention prioritizes nearby and reliable actors, and the attended embedding is fused with ego motion, route progress, and lane features into a compact policy state.

### 2. Dense differentiable reward
The reward combines four objectives — **Safety**, **Progress**, **Comfort**, and **Uncertainty** — producing smoother optimization signals than purely event-triggered penalties and improving stability across source and target domains.

### 3. Uncertainty-gated exploration
Decision-time uncertainty is decomposed into aleatoric uncertainty (per-edge heteroscedastic heads) and epistemic uncertainty (critic ensemble). These are merged into a confidence signal that gates entropy, making the policy more cautious under low confidence.

### 4. Causal-semantic transfer
The transfer objective aligns action distributions, relational attention patterns, and uncertainty statistics, combined with MAML-style initialization for few-shot adaptation to new towns and weather conditions.

---

## Simulation Screenshots

<p align="center">
  <img src="./screenshot/1.png" width="32%" alt="Screenshot 1"/>
  &nbsp;
  <img src="./screenshot/2.png" width="32%" alt="Screenshot 2"/>
  &nbsp;
  <img src="./screenshot/3.png" width="32%" alt="Screenshot 3"/>
</p>

---

## Demo Videos

> Click a thumbnail to open the video.

<p align="center">
  <a href="./video/1.mp4"><img src="./screenshot/1.png" width="30%" alt="Demo 1"/></a>
  &nbsp;&nbsp;
  <a href="./video/2.mp4"><img src="./screenshot/2.png" width="30%" alt="Demo 2"/></a>
  &nbsp;&nbsp;
  <a href="./video/3.mp4"><img src="./screenshot/3.png" width="30%" alt="Demo 3"/></a>
</p>
<p align="center">
  <a href="./video/1.mp4">Demo 1</a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="./video/2.mp4">Demo 2</a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="./video/3.mp4">Demo 3</a>
</p>

---

## Graphs and Visual Results

<p align="center">
  <img src="./graphs/reward_comparison.png" width="48%" alt="Reward Comparison"/>
  &nbsp;
  <img src="./graphs/state_route.png" width="48%" alt="State Route Analysis"/>
</p>
<p align="center">
  <img src="./graphs/state_stability.png" width="48%" alt="State Stability Analysis"/>
  &nbsp;
  <img src="./graphs/uncertainty_metrics.png" width="48%" alt="Uncertainty Metrics"/>
</p>

---

## Repository Layout

```text
safe-driving-drl/
├── car.py
├── README.md
├── requirements.txt
├── close_loop.png
├── checkpoints/
│   ├── source_agent_best.pt
│   └── source_agent.pt
├── diagrams/
│   └── unified_framework.png
├── docs/
│   └── PROJECT_NOTES.md
├── graphs/
│   ├── reward_comparison.png
│   ├── state_route.png
│   ├── state_stability.png
│   └── uncertainty_metrics.png
├── logs/
│   ├── eval_500m_log.txt
│   ├── eval_500m_npc_log.txt
│   ├── eval_town02_150m_npc_log.txt
│   ├── eval_town02_200m_npc_log.txt
│   ├── eval_town05_500m_npc_log.txt
│   ├── eval_town05_npc_log.txt
│   ├── eval_town10_mixed_log.txt
│   └── log.txt
├── results/
│   ├── summaries/
│   │   ├── Town02_zeroshot_source_agent_summary.csv
│   │   ├── Town05_zeroshot_source_agent_summary.csv
│   │   └── Town10HD_Opt_zeroshot_source_agent_summary.csv
│   ├── Town02_zeroshot_source_agent.csv
│   ├── Town05_zeroshot_source_agent.csv
│   ├── Town10HD_Opt_zeroshot_source_agent.csv
│   └── trajectories/
├── screenshot/
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
├── scripts/
│   ├── eval_town05_300m_npc.sh
│   ├── eval_town05_500m_npc.sh
│   ├── eval_town05.sh
│   ├── eval_town10hd_500m_npc.sh
│   ├── eval_town10hd.sh
│   ├── run_carla_server.sh
│   ├── show_tree.sh
│   ├── train_paper.sh
│   └── train_short.sh
└── video/
    ├── 1.mp4
    ├── 2.mp4
    └── 3.mp4
```

---

## Requirements

- Ubuntu 20.04 / 22.04
- CARLA 0.9.15
- Python 3.10
- PyTorch 2.x

```bash
conda create -n safe python=3.10
conda activate safe
pip install -r requirements.txt
```

Example `requirements.txt`:
```text
torch>=2.0
numpy
gymnasium
matplotlib
```

---

## Quick Start

### 1. Start CARLA

```bash
# Headless mode
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2200 &
sleep 15
```

Or with GUI:

```bash
./CarlaUE4.sh -opengl -quality-level=Low -windowed \
  -ResX=800 -ResY=600 -carla-rpc-port=2200 -nosound &
sleep 15
```

### 2. Train the source policy

```bash
python3 -u car.py \
  --mode train \
  --host localhost --port 2200 --tm-port 8001 \
  --train-town Town10HD_Opt \
  --spawn-index -1 \
  --train-goal-index -1 \
  --train-steps 500000 \
  --train-npc-min 8 --train-npc-max 20 \
  --source-weather night_rain_fog \
  --out-dir ./culrt_carla_0915_aligned \
  --maml-warmup-batches 10 \
  --start-steps 2000 --update-after 1000 \
  --save-every-steps 5000 \
  --debug 2>&1 | tee ./culrt_carla_0915_aligned/train_log.txt
```

### 3. Evaluate on Town05

```bash
python3 -u car.py \
  --mode eval \
  --host localhost --port 2200 \
  --target-town Town05 \
  --target-weather mixed \
  --spawn-index 0 \
  --eval-episodes 20 \
  --checkpoint ./culrt_carla_0915_aligned/models/source_agent.pt \
  --out-dir ./culrt_carla_0915_aligned \
  --target-goal-index -1 \
  --npc-min 8 --npc-max 15 \
  --debug
```

### 4. Evaluate on Town10HD with a longer route

```bash
python3 -u car.py \
  --mode eval \
  --host localhost --port 2200 \
  --target-town Town10HD_Opt \
  --target-weather mixed \
  --spawn-index -1 \
  --eval-episodes 20 \
  --checkpoint ./culrt_carla_0915_aligned/models/source_agent.pt \
  --out-dir ./culrt_carla_0915_aligned \
  --target-goal-index -1 \
  --route-target-length 500 \
  --npc-min 8 --npc-max 15 \
  --no-rendering \
  --debug 2>&1 | tee ./culrt_carla_0915_aligned/eval_500m_npc_log.txt
```

### 5. Evaluate cross-town transfer on Town02

```bash
python3 -u car.py \
  --mode eval \
  --host localhost --port 2200 \
  --target-town Town02 \
  --target-weather mixed \
  --spawn-index 0 \
  --eval-episodes 20 \
  --checkpoint ./culrt_carla_0915_aligned/models/source_agent.pt \
  --out-dir ./culrt_carla_0915_aligned \
  --target-goal-index -1 \
  --npc-min 8 --npc-max 15 \
  --debug
```

---

## Key Flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `eval` | `train` / `eval` / `adapt` / `policy` |
| `--train-town` | `Town10HD_Opt` | Source training map |
| `--target-town` | `Town02` | Evaluation or transfer target map |
| `--spawn-index` | `0` | `-1` for random spawn per run |
| `--route-target-length` | `200` | Route length in meters |
| `--npc-min` / `--npc-max` | `0` / `2` | NPC traffic count range |
| `--no-rendering` | off | Disable CARLA GUI |
| `--source-weather` | `night_rain_fog` | Source-domain weather |
| `--target-weather` | `mixed` | Target-domain weather |

---

## Troubleshooting

**`import carla` fails**
```bash
export CARLA_ROOT=~/CARLA_0.9.15
export PYTHONPATH=$PYTHONPATH:~/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg
```

**CARLA server crashes with many NPCs** — Add `--no-rendering`

**Town02 route planner warning** — Reduce route length: `--route-target-length 150`

**Alpha collapses during training** — This implementation includes a clamp on `log_alpha` to prevent entropy collapse during long training runs.

---

## Citation

```bibtex
@inproceedings{uddin2026reliable,
  title     = {Reliable Policy Transfer for Safety-Aware End-to-End Driving with Deep Reinforcement Learning},
  author    = {Uddin, Md. Borhan and Raza, Arif and Lin, Zhiliang and Wang, Lu and Li, Jianqiang and Chen, Jie},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgements

This work was supported by the National Engineering Laboratory for Big Data System Computing Technology, Shenzhen University.
