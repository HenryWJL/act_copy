import torch
import os
import h5py
from glob import glob

import IPython
e = IPython.embed

def get_norm_stats(args):
    all_qpos_data = []
    all_action_data = []
    file_paths = glob(os.path.join(args.dataset_dir, '*.hdf5'))
    for path in file_paths:
        with h5py.File(path, 'r') as f:
            qpos = f['/observations/qpos'][()]
            action = f['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy()}

    return stats
