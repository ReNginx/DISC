# DISC: Distilled HyLaP Release

Minimal public release of **HyLaP** (Hyper Language-Action Policy) — a hypernetwork-based
policy that conditions on a language instruction to generate the action-decoder weights
for a given task.

This repo contains only the HyLaP model and the LIBERO / metaworld pipelines. Baseline
methods (Diffusion Policy, BAKU, HyperZero, HyPoGen, DiT, VQ-BeT, …) and ablation
configs from the research codebase are intentionally excluded.

## What's in here

```
DISC/
├── model/
│   ├── hylap/                # HyLaP (HyPoGen2) policy
│   ├── image_encoders.py
│   └── language_encoders.py
├── dataloader/               # LIBERO-original + metaworld data modules
├── trainer/                  # GenericTrainer + HyLaPTrainer (Lightning)
├── evaluator/                # LIBERO + metaworld evaluators (with TTT support)
├── utils/                    # Module-size helpers used by HyLaP
├── config/                   # Hydra configs
│   ├── default.yaml
│   ├── hylap.yaml            # main HyLaP training entry
│   ├── libero_90{,_eval,_ttt}.yaml
│   ├── metaworld{,_eval,_ttt}.yaml
│   ├── data/                 # libero_90 / spatial / goal / object / 10 (long) / metaworld
│   ├── model/{defaults,hylap}.yaml
│   ├── trainer/default.yaml
│   └── logger/{tensorboard,wandb}.yaml
├── scripts/hylap/            # convenience training & evaluation shell scripts
└── train.py                  # main entrypoint (Hydra)
```

Supported task suites:

- **LIBERO**: `libero_90`, `libero_spatial`, `libero_goal`, `libero_object`, `libero_10` (LIBERO-Long)
- **metaworld**

## Installation

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git modified_libero
cd ..

uv sync
touch ./third_party/modified_libero/libero/__init__.py
uv pip install -e ./third_party/modified_libero

bash utils/post_install.sh
```

## Training

Train HyLaP on LIBERO-90 (default):

```bash
python train.py --config-name=hylap
```

Train on a different LIBERO suite or on metaworld:

```bash
python train.py --config-name=hylap data=libero_spatial
python train.py --config-name=hylap data=libero_goal
python train.py --config-name=hylap data=libero_object
python train.py --config-name=hylap data=libero_10        # LIBERO-Long
python train.py --config-name=hylap data=metaworld
```

Override common hyperparameters from the CLI:

```bash
python train.py --config-name=hylap model.optimizer.lr=5e-4 data.batch_size=64
```

## Evaluation

```bash
python train.py --config-name=libero_90_eval model=hylap +ckpt_path=path/to/ckpt.ckpt
python train.py --config-name=metaworld_eval model=hylap +ckpt_path=path/to/ckpt.ckpt
```

## Test-Time Training (TTT)

```bash
python train.py --config-name=libero_90_ttt   +ckpt_path=path/to/ckpt.ckpt
python train.py --config-name=metaworld_ttt   +ckpt_path=path/to/ckpt.ckpt
```

Convenience shell scripts live under [scripts/hylap/](scripts/hylap/).
