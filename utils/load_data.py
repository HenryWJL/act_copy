import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from pytorch3d.ops import knn_points, knn_gather

from .acronym import load_mesh, load_grasps


class Dataset(Dataset):
    
    
    def __init__(self, args):
        super().__init__()
        
        self.object_fname = glob(os.path.join(object_dir, mode, "**.h5"))
        self.mesh_dir = os.path.join(mesh_dir, mode)
        self.point_num = point_num
    
    
    def __getitem__(self, key):
        T, success = load_grasps(self.object_fname[key])
        mesh = load_mesh(self.object_fname[key], mesh_root_dir=self.mesh_dir)
        point_cloud = mesh.sample(self.point_num)
        point_cloud = torch.from_numpy(point_cloud).float()
        return point_cloud, T, success
    
    
    def __len__(self):
        return len(self.object_fname)


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim
