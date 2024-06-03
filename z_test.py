import torch
import h5py
import numpy as np
from pathlib import Path

dir = Path('/home/wangjunlin/project/act_pytorch/data/cup_place')
for p in list(dir.glob("*.h5")):
    with h5py.File(str(p), 'r') as f:
        data = f['/observations/images/head_camera'][:]
        if type(data) != np.ndarray:
            print(p.name)