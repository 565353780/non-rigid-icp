# Non Rigid ICP

## Setup

```bash
conda create -n nricp python=3.11
conda activate nricp
./setup.sh
```

## Run Case1 Watertight Fitting

The current production/research entrypoint for fitting the watertight case1 mesh
to the original target mesh is:

```bash
cd /home/lichanghao/github/Watertight/non-rigid-icp
CUDA_VISIBLE_DEVICES=2 /vepfs-cnbja62d5d769987/lichanghao/miniconda3/envs/flux/bin/python -m non_rigid_icp.Test.exp_case1_front_advance_locked
```

This script runs the locked front-advancing refine algorithm:

- optimized vertices are locked after each step and are never moved again;
- only newly inserted subdivision vertices are optimized in later steps;
- subdivision is applied synchronously to the fitted mesh and the source reference
  mesh, preserving per-vertex correspondence;
- exact penetration relaxation is applied each step to avoid introducing new
  through-penetrations.

Outputs are written to:

```text
output/case1_front_advance_locked/
```

Key files:

- `front_advance_locked_log.json`: initial/per-step metrics and refinement log.
- `step_00.ply` ... `step_03.ply`: full fitted meshes per step.
- `debug/bbox_*/initial_source_crop.ply`: initial source crop for each bbox.
- `debug/bbox_*/step_XX_crop.ply`: per-step fitted crop for each bbox.
- `debug/bbox_*/target_crop.ply`: target crop for each bbox.

Common runtime parameters can be overridden with environment variables:

```bash
N_STEPS=4 \
RELAX_BACKOFF=0.8 \
RELAX_ITERS=60 \
REFINE_MEAN_MULT=1.0 \
MAX_REFINE_FACES=1500000 \
CUDA_VISIBLE_DEVICES=2 \
/vepfs-cnbja62d5d769987/lichanghao/miniconda3/envs/flux/bin/python -m non_rigid_icp.Test.exp_case1_front_advance_locked
```

## Legacy Demo

```bash
python demo.py
```

## Visualization

```bash
tensorboard --logdir ./logs --host 0.0.0.0 --port 6006 --samples_per_plugin images=100
```

and then open

```bash
127.0.0.1:6006
```

for visualization

## Enjoy it~
