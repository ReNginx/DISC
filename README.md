# DISC: Decoupling Instruction from State-Conditioned Control via Policy Generation

The offiical repository of RSS 2026 paper "DISC: Decoupling Instruction from State-Conditioned Control via Policy Generation"

Supported task suites:

- **LIBERO**: `libero_90`, `libero_spatial`, `libero_goal`, `libero_object`, `libero_10` 
- **Metaworld(WIP)**

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

## Data Preparation

We use the openvla version of libero (which has resolution of 256x256 and filters out no-op actions) in our experiments.

You can download libero-10, libero-goal, libero-object, libero-spatial from 'https://huggingface.co/datasets/openvla/modified_libero_rlds'

For libero-90, you need to download the official version of libero-90 from https://github.com/Lifelong-Robot-Learning/LIBERO then convert it into openvla version using the script provided by the openvla repository.

Once you download the data, you should organize the files into following data format.

/data/libero/{libero_90|libero_object|libero_spatial|libero_10|libero_goal}

## Training

Train HyLaP on LIBERO-90 (default):

```bash
python train.py --config-name=hylap
```

Train on a different LIBERO suite:

```bash
python train.py --config-name=hylap data=libero_spatial
python train.py --config-name=hylap data=libero_goal
python train.py --config-name=hylap data=libero_object
python train.py --config-name=hylap data=libero_10
```

## Evaluation

```bash
python train.py --config-name=libero_90_eval model=hylap +ckpt_path=path/to/ckpt.ckpt
```

## Test-Time Training (TTT)

```bash
python train.py --config-name=libero_90_ttt   +ckpt_path=path/to/ckpt.ckpt
```

## Citation 
If you find this work helpful in your reserch, consider citing 
```
@inproceedings{
ren2026disc,
title={{DISC}: Decoupling Instruction from State-Conditioned Control via Policy Generation},
author={Hanxiang Ren, Pei Zhou, Xunzhe Zhou, Yanchao Yang},
booktitle={Robotics: Science and Systems 2026},
year={2026},
url={https://openreview.net/forum?id=i9ynkagJuj}
}
```
