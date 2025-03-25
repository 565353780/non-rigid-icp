# Non Rigid ICP

## Setup

```bash
conda create -n nricp python=3.11
conda activate nricp
./setup.sh
```

## Run

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
