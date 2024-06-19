This is a PyTorch implementation of the imitation learning algorithm ACT proposed in _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_. 
- **Original codebase:** [act](https://github.com/tonyzhaozh/act).
- **Paper:** [here](https://arxiv.org/abs/2304.13705).

## Usage
### 1. Collect human demonstrations.
```bash
cd utils
python collect_data.py
```
### 2. Train the policy.
```bash
python train.py --dataset_dir <YOUR_DATASET_ROOT>
```
### 3. Evaluate the policy.
```bash
python test.py --checkpoint <YOUR_CHECKPOINT_PATH>
```
