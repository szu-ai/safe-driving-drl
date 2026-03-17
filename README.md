# Reliable E2E Policy Transfer with DRL (CARLA Project)

This repository packages the code, checkpoints, logs, paper PDF, and reproducibility scripts for the paper project **“Reliable Policy Transfer for Safety-Aware End-to-End Driving with Deep Reinforcement Learning.”**

## Included contents

- `car.py` — main training and evaluation script
- `paper/` — uploaded paper PDF
- `checkpoints/` — trained source checkpoints
- `results/` — evaluation CSV outputs and summaries
- `results/trajectories/` — per-episode trajectory CSV files
- `logs/` — training and evaluation logs
- `scripts/` — ready-to-run shell commands for CARLA, training, and evaluation

## Paper-aligned setup

The uploaded paper describes a CARLA **0.9.15** closed-loop evaluation setup with:

- synchronous stepping at **20 Hz**
- Tesla Model 3 ego vehicle
- SAC-based control with uncertainty-gated exploration
- adverse-weather source training in **Town10HD_Opt / Town10HD**
- cross-town evaluation and transfer to **Town02** and **Town05**

## Quick start

### 1) Start CARLA

```bash
bash scripts/run_carla_server.sh
```

### 2) Train the source policy

```bash
bash scripts/train_paper.sh
```

### 3) Run evaluation

```bash
bash scripts/eval_town05.sh
bash scripts/eval_town10hd.sh
```

## Reproducibility scripts

- `scripts/run_carla_server.sh`
- `scripts/train_paper.sh`
- `scripts/train_short.sh`
- `scripts/eval_town05.sh`
- `scripts/eval_town10hd.sh`
- `scripts/eval_town10hd_500m_npc.sh`
- `scripts/eval_town05_500m_npc.sh`
- `scripts/eval_town05_300m_npc.sh`

## Result snapshots

| Town | Success Rate (%) | Driving Score | IS | Coll./km | Off/km | TO/km | Dist. (km) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Town02 | 75.0 | 70.50 | 0.75 | 0.3733 | 0.0000 | 1.4934 | 2.6785 |
| Town05 | 100.0 | 94.56 | 1.00 | 0.0000 | 0.0000 | 0.0000 | 4.4746 |
| Town10HD_Opt | 70.0 | 66.52 | 0.70 | 0.5586 | 0.1862 | 0.3724 | 5.3703 |

## Notes

- The 500 m NPC evaluation scripts use **distinct log filenames** to avoid accidental overwrite.
- The project keeps your original `car.py` unchanged and organizes generated assets into clearer folders.
- If `import carla` fails, either install the CARLA Python API in your environment or export `CARLA_ROOT` to the CARLA 0.9.15 installation.

## Suggested environment

- Python 3.10
- CARLA 0.9.15
- PyTorch
- NumPy
- Gym or Gymnasium

## Repository layout

```text
reliable-e2e-policy-transfer-drl-carla/
├── car.py
├── README.md
├── requirements.txt
├── .gitignore
├── paper/
├── checkpoints/
├── logs/
├── results/
│   ├── summaries/
│   └── trajectories/
└── scripts/
```
