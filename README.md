# DISC: Decoupling Instruction from State-Conditioned Control via Policy Generation

Official implementation of the RSS 2026 paper:

**DISC: Decoupling Instruction from State-Conditioned Control via Policy Generation**

> Hanxiang Ren, Pei Zhou, Xunzhe Zhou, and Yanchao Yang  
> Robotics: Science and Systems (RSS), 2026

## 🔍 Overview

DISC is a policy-generation framework that decouples high-level language instructions from state-conditioned low-level control. This repository provides training, evaluation, and test-time training code for reproducing the results reported in the paper.

## 🧩 Supported Task Suites

- **LIBERO**
  - `libero_90`
  - `libero_spatial`
  - `libero_goal`
  - `libero_object`
  - `libero_10`
- **Meta-World**: work in progress

## ⚙️ Installation

Clone and install the modified LIBERO dependency:

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

## 📦 Data Preparation

We use the OpenVLA version of LIBERO in our experiments. This version uses `256 x 256` image observations and filters out no-op actions.

Please download the official LIBERO dataset from the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO), then convert it to the OpenVLA format using [`regenerate_libero_dataset.py`][openvla-regenerate-libero] from the OpenVLA repository.

After downloading and converting the datasets, organize them as follows:

```text
/data/libero/
├── libero_90/
├── libero_object/
├── libero_spatial/
├── libero_10/
└── libero_goal/
```

## 🚀 Training

Train DISC on LIBERO-90:

```bash
python train.py --config-name=hylap
```

Train on other LIBERO suites:

```bash
python train.py --config-name=hylap data=libero_spatial
python train.py --config-name=hylap data=libero_goal
python train.py --config-name=hylap data=libero_object
python train.py --config-name=hylap data=libero_10
```

## 📊 Evaluation
Our checkpoints can be downloaded from [DISC Huggingface Repo](https://huggingface.co/ReNginx/DISC).

Evaluate a trained checkpoint on LIBERO-90:

```bash
python train.py --config-name=libero_90_eval model=hylap +ckpt_path=/path/to/checkpoint.ckpt
```

## 🔁 Test-Time Training

Run test-time training (TTT) on LIBERO-90:

```bash
python train.py --config-name=libero_90_ttt +ckpt_path=/path/to/checkpoint.ckpt
```

## 📝 Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@inproceedings{ren2026disc,
  title={{DISC}: Decoupling Instruction from State-Conditioned Control via Policy Generation},
  author={Hanxiang Ren and Pei Zhou and Xunzhe Zhou and Yanchao Yang},
  booktitle={Robotics: Science and Systems},
  year={2026},
  url={https://openreview.net/forum?id=i9ynkagJuj}
}
```

## 🙏 Acknowledgements

This codebase builds on resources from the LIBERO, OpenVLA, and [HyPoGen](https://github.com/ReNginx/HyPoGen) projects. We thank the authors for releasing their datasets, environments, code, and tools.

[openvla-libero-rlds]: https://huggingface.co/datasets/openvla/modified_libero_rlds
[openvla-regenerate-libero]: https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py
