# Project Packaging Notes

This zip was organized from the uploaded `car.py`, uploaded project archive, and uploaded paper PDF.

Changes made for packaging only:

1. Grouped checkpoints under `checkpoints/`.
2. Grouped summary CSVs under `results/summaries/`.
3. Grouped per-episode trajectory CSVs under `results/trajectories/`.
4. Grouped text logs under `logs/`.
5. Added reproducibility shell scripts under `scripts/`.
6. Standardized long-run 500 m NPC log filenames so different town evaluations do not overwrite each other.

The training/evaluation logic in `car.py` was not modified.
