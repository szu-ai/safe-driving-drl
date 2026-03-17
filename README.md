# Reliable Policy Transfer for Safety-Aware End-to-End Driving with Deep Reinforcement Learning

**CVPR 2026 · Submission 38759**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange)](https://carla.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Md. Borhan Uddin, Arif Raza, Zhiliang Lin, Lu Wang, Jianqiang Li, Jie Chen  
> College of Computer Science and Software Engineering, Shenzhen University  
> Corresponding author: chenjie@szu.edu.cn

---

## Overview

This repository contains the official implementation of a unified Deep Reinforcement Learning (DRL) framework for safety-aware end-to-end autonomous driving with reliable policy transfer. The framework addresses four key challenges in closed-loop driving:

1. **Ego-centric relational state** — an uncertainty-weighted attention graph that encodes causal interactions between the ego vehicle and nearby agents, making safety-critical influences explicit to the policy.

2. **Differentiable multi-objective reward** — smooth surrogates for safety, progress, comfort, and uncertainty that replace sparse event-driven penalties, stabilizing gradients across domains.

3. **Uncertainty-gated exploration** — joint aleatoric-epistemic estimation from per-edge heteroscedastic variance and a critic ensemble, compressed into a calibrated confidence signal σ̄ that modulates SAC policy entropy for risk-aware exploration.

4. **Causal-semantic policy transfer** — a transfer objective aligning action distributions (KL), relational attention (MMD), and uncertainty statistics across domains, paired with MAML-style initialization for few-shot adaptation.

---

## Key Results

Closed-loop evaluation in CARLA 0.9.15 across Town10HD (source), Town02, and Town05 (targets) with 8–20 NPC vehicles under adverse weather:

| Map | SR (%) | DS | IS | Coll./km | Off/km | TO/km |
|-----|-------:|----:|----:|---------:|-------:|------:|
| Town10HD (source training) | 91.2 | 94.1 | 1.00 | 0.000 | 0.000 | 0.000 |
| Town05 (zero-shot transfer) | 100.0 | 94.1 | 1.00 | 0.000 | 0.000 | 0.000 |
| Town02 — Policy Learning | — | 188.6 | 0.88 | 0.007 | 0.005 | 0.003 |
| Town02 — Source Domain | — | 205.7 | 0.92 | 0.006 | 0.004 | 0.002 |
| Town02 — **Target (full transfer)** | — | **214.3** | **0.94** | **0.005** | **0.003** | **0.001** |

**SR** = Success Rate, **DS** = Driving Score, **IS** = Infraction Score.  
Town05 zero-shot result: mean CTE = 0.183 m, mean heading error = 0.021 rad.

---

## Architecture

```
Perception (CARLA sensor stack)
        │
        ▼
Ego-Centric Relational Graph          ← Eq. (6–8)
  edges: [Δp, Δv, class, κ, σ²_ale]
  attention: α_i = softmax(−‖Δp_i‖² / (σ²_i + ε))
  state: s_t = [z_t; v_ego; a^{t−1}; d_goal; φ_lane; σ²_ale]
        │
        ▼
SAC Actor-Critic                       ← Eq. (14–15)
  uncertainty: σ²_dec = σ²_ale + σ²_epi  (critic ensemble)
  entropy gate: β(σ̄) = β₀(1 − σ̄)
        │
        ├─── Dense Reward               ← Eq. (9–13)
        │    r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u
        │
        └─── Transfer (source→target)  ← Eq. (16–18)
             L_trans = L_KL + λ_α MMD(α_s, α_t) + λ_u ‖u_s − u_t‖²
             + MAML initialization (Eq. 17)
```

---

## Repository Layout

```
reliable-e2e-policy-transfer-drl-carla/
├── car.py                    ← single-file framework (train / eval / adapt)
├── README.md
├── requirements.txt
├── .gitignore
├── paper/
│   └── Reliable_E2E_Policy_Transfer_with_DRL.pdf
├── checkpoints/
│   ├── source_agent.pt       ← trained source policy (Town10HD, 500k steps)
│   └── source_agent_best.pt  ← best checkpoint by episode success
├── results/
│   ├── Town02_zeroshot_source_agent.csv
│   ├── Town05_zeroshot_source_agent.csv
│   ├── Town10HD_Opt_zeroshot_source_agent.csv
│   ├── summaries/            ← per-run aggregate metrics
│   └── trajectories/         ← per-episode x,y trajectory CSVs (ep01–ep20)
├── logs/                     ← training and evaluation terminal logs
└── scripts/
    ├── run_carla_server.sh
    ├── train_paper.sh         ← full 500k-step source training
    ├── train_short.sh         ← quick 50k-step smoke test
    ├── eval_town05.sh
    ├── eval_town10hd.sh
    ├── eval_town10hd_500m_npc.sh
    ├── eval_town05_500m_npc.sh
    └── eval_town05_300m_npc.sh
```

---

## Requirements

- Ubuntu 20.04 / 22.04
- CARLA 0.9.15 ([download](https://github.com/carla-simulator/carla/releases/tag/0.9.15))
- Python 3.10
- PyTorch ≥ 2.0

```bash
conda create -n safe python=3.10
conda activate safe
pip install -r requirements.txt
```

`requirements.txt` contents:
```
torch>=2.0
numpy
gymnasium
matplotlib
```

---

## Quick Start

### 1. Start CARLA

```bash
# Headless (recommended — avoids GPU crash with many NPCs)
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2200 &

# Or with GUI at low quality
./CarlaUE4.sh -opengl -quality-level=Low -windowed \
  -ResX=800 -ResY=600 -carla-rpc-port=2200 -nosound &

sleep 15   # wait for server to be ready
```

### 2. Train the source policy (Town10HD, 500k steps)

Full paper training:
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

Quick smoke test (50k steps, ~2–4 hours on CPU):
```bash
python3 -u car.py \
  --mode train \
  --host localhost --port 2200 --tm-port 8001 \
  --train-town Town10HD_Opt \
  --spawn-index -1 \
  --train-goal-index -1 \
  --train-steps 50000 \
  --train-npc-min 8 --train-npc-max 20 \
  --source-weather night_rain_fog \
  --out-dir ./culrt_carla_0915_aligned \
  --maml-warmup-batches 10 \
  --debug 2>&1 | tee ./culrt_carla_0915_aligned/train_log2.txt
```

### 3. Evaluate — Town05 (zero-shot)

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

### 4. Evaluate — Town10HD (500m route, with NPC traffic)

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

### 5. Evaluate — Town02 (cross-town transfer)

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
|------|---------|-------------|
| `--mode` | `eval` | `train` / `eval` / `adapt` / `policy` |
| `--train-town` | `Town10HD_Opt` | Source training map |
| `--target-town` | `Town02` | Evaluation / transfer target map |
| `--spawn-index` | `0` | `-1` = random spawn per episode (recommended) |
| `--route-target-length` | `200` | Override route length in metres (e.g. `500` for longer trajectories) |
| `--npc-min` / `--npc-max` | `0` / `2` | NPC vehicle count range per episode |
| `--no-rendering` | off | Disable CARLA GUI (prevents GPU crash with many NPCs) |
| `--source-weather` | `night_rain_fog` | Training weather regime |
| `--target-weather` | `mixed` | Evaluation weather regime |

---

## Method Details

### Ego-centric relational state (Sec. 3.3)
Each scene entity contributes an edge `e_i = [Δp_i, Δv_i, c_i, κ_i, σ²_i]` into the ego node. Uncertainty-weighted attention `α_i = softmax(−‖Δp_i‖² / (σ²_i + ε))` prioritises nearby, confident actors. The attended embedding is fused with ego dynamics and lane features into compact state `s_t`.

### Dense reward (Sec. 3.4)
`r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u` with weights summing to 1. Safety `r_s` uses differentiable lane-barrier, proximity, and red-light surrogates. Progress `r_p = tanh(Δs/τ_s)`. Comfort `r_c = −κ_j j²_t − κ_δ δ̇²_t`. Uncertainty `r_u = 1 − σ̄`.

### Uncertainty-gated SAC (Sec. 3.5)
`σ²_dec = σ²_ale + σ²_epi`. Aleatoric `σ²_ale` comes from per-edge heteroscedastic heads. Epistemic `σ²_epi = Var_k[Q_k(s,a)]` from a 5-critic ensemble. Entropy gate: `β(σ̄) = β₀(1 − σ̄)` reduces exploration under low confidence.

### Causal-uncertainty transfer (Sec. 3.6)
`L_trans = L_KL + λ_α MMD(α_s, α_t) + λ_u ‖u_s − u_t‖²` aligns action distributions, attention patterns, and uncertainty statistics. MAML-style initialization enables few-shot target adaptation.

---

## Troubleshooting

**`import carla` fails:**
```bash
export CARLA_ROOT=~/CARLA_0.9.15
# or add the egg to PYTHONPATH:
export PYTHONPATH=$PYTHONPATH:~/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg
```

**CARLA server crashes with many NPCs:**
Add `--no-rendering` to the eval command. This disables the GUI entirely but all results and CSVs are still generated.

**`[WARN] planner auto-route unavailable` on Town02:**
Normal for small maps. The fallback local lane-follow route is valid. Use `--route-target-length 150` to reduce the target length to fit Town02's road segments.

**Alpha collapses to 0.0 during training:**
Fixed in this implementation via `log_alpha.clamp_(min=math.log(0.01))` in `SACAgent.update()`. Without this fix, alpha hits zero at ~18k steps causing 97% collision rate.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{uddin2026reliable,
  title     = {Reliable Policy Transfer for Safety-Aware End-to-End Driving
               with Deep Reinforcement Learning},
  author    = {Uddin, Md. Borhan and Raza, Arif and Lin, Zhiliang and
               Wang, Lu and Li, Jianqiang and Chen, Jie},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgements

This work was supported by the National Engineering Laboratory for Big Data System Computing Technology, Shenzhen University.
