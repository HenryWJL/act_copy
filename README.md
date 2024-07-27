This is the PyTorch implementation of [_Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_](https://arxiv.org/abs/2304.13705). Code in this repository is copied from the [official codebase](https://github.com/tonyzhaozh/act) and simplified such that minimal requirements are demanded. Also, sufficient annotations are added to help users better understand the algorithm.

## Installation
Create a conda environment:
```bash
conda env create -f conda_env.yaml
conda activate act
```

## Usage
### 1. Collect Human Demonstrations.
```bash
python utils/collect_data.py
```
### 2. Train ACT Policy.
```bash
python train.py --dataset_dir <YOUR_DATASET_ROOT>
```
### 3. Evaluate ACT policy.
```bash
python eval.py --checkpoint <YOUR_CHECKPOINT_PATH>
```
