import os
import cv2
import h5py
import torch
import numpy as np
from pathlib import Path
from glob import glob
from einops import rearrange
from typing import Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader

from .normalize import get_norm_stats

import IPython
e = IPython.embed


# class ACTDataset(Dataset):
    
#     def __init__(self, args) -> None:
#         super().__init__()
#         self.dataset_dir = Path(args.dataset_dir)
#         self.file_paths = list(self.dataset_dir.glob('*.h5')) \
#             + list(self.dataset_dir.glob('*.hdf5'))
#         self.cameras = args.cameras
#         self.action_horizon = args.action_horizon
#         self.norm_mode = args.norm_mode
#         """
#         norm_mode: "max" or "mean_std".
#             "max": data / data_max
#             "mean_std": (data - mean) / std
#         """
#         self.images = None
#         self.proprios = None
#         self.actions = None
#         self.is_pads = None
#         self.norm_dict = dict.fromkeys(["proprio", "action"])
#         self.get_norm_dict()
#         self.load()

#     def __getitem__(self, idx):
#         return self.images[idx], self.proprios[idx],\
#             self.actions[idx], self.is_pads[idx]

#     def __len__(self) -> int:
#         return int(self.proprios.shape[0])
    
#     def normalize(
#         self,
#         data: np.ndarray,
#         data_type: Optional[str] = "proprio"
#     ) -> np.ndarray:
#         """
#         Args:
#             data: (N, D)
            
#             data_type: "proprio" or "action"
                
#         Returns:
#             normalized data (N, D)
#         """
#         if self.norm_mode == "max":
#             data_max = self.norm_dict[data_type]["max"]
#             data_norm = data / data_max
#         elif self.norm_mode == "mean_std":
#             data_mean = self.norm_dict[data_type]["mean"]
#             data_std = self.norm_dict[data_type]["std"]
#             data_norm = (data - data_mean) / data_std
#         return data_norm
    
#     def get_norm_params(
#         self,
#         data: np.ndarray
#     ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
#         """
#         Args:
#             data: (N, D)
                
#         Returns:
#             maximum values (1, D) or
#             mean (1, D) and standard deviation (1, D)
#         """
#         if self.norm_mode == "max":
#             maxima = data.max(axis=0, keepdims=True)
#             return maxima
#         elif self.norm_mode == "mean_std":
#             mean = data.mean(axis=0, keepdims=True)
#             std = data.std(axis=0, keepdims=True)
#             std = np.clip(std, 1e-2, np.inf)
#             return (mean, std)
#         else:
#             raise NotImplementedError
    
#     def get_norm_dict(self) -> None:
#         proprios = list()
#         actions = list()
#         for path in self.file_paths:
#             with h5py.File(path, 'r') as f:
#                 action = f['/action'][()]  # (episode_len, action_dim)
#                 proprio = f['/observations/qpos'][()]  # (episode_len, pos_dim)
#                 actions.append(action)
#                 proprios.append(proprio)
#         proprios = np.concatenate(proprios, axis=0)
#         actions = np.concatenate(actions, axis=0)
#         proprio_data = self.get_norm_params(proprios)
#         action_data = self.get_norm_params(actions)
#         if self.norm_mode == "max":
#             self.norm_dict["proprio"] = {"max": proprio_data}
#             self.norm_dict["action"] = {"max": action_data}
#         elif self.norm_mode == "mean_std":
#             proprio_mean, proprio_std = proprio_data
#             action_mean, action_std = action_data
#             self.norm_dict["proprio"] = {"mean": proprio_mean, "std": proprio_std}
#             self.norm_dict["action"] = {"mean": action_mean, "std": action_std}
    
#     def load(self):
#         images = list()
#         proprios = list()
#         actions = list()
#         is_pads = list()
#         for path in self.file_paths:
#             with h5py.File(path, 'r') as f:
#                 action = f['/action'][()]  # (episode_len, action_dim)
#                 proprio = f['/observations/qpos'][()]  # (episode_len, pos_dim)
#                 multi_image = [
#                     f[f'/observations/images/{camera}'][()]
#                     for camera in self.cameras
#                 ]
#                 episode_len = action.shape[0]  # the horizon of this episode
#                 # decode images (if necessary)
#                 if len(multi_image[0].shape) != 4:
#                     for idx, image in enumerate(multi_image):
#                         image_decode = [
#                             cv2.imdecode(image[i], 1)
#                             for i in range(image.shape[0])
#                         ]
#                         multi_image[idx] = np.stack(image_decode, axis=0)  # (episode_len, h, w, c)    
#                 # stack images from different camera views
#                 image = np.stack(multi_image, axis=1)  # (episode_len, num_camera, h, w, c)
#                 # normalize proprioceptions and actions
#                 proprio = self.normalize(proprio, "proprio")
#                 action = self.normalize(action, "action")     
#                 # zero pad actions
#                 zero_pad = np.zeros(
#                     (self.action_horizon - 1, action.shape[1]),
#                     dtype=np.float32
#                 )
#                 action = np.concatenate([action, zero_pad], axis=0)  
#                 # denote the positions of zero paddings      
#                 is_pad = np.zeros(action.shape[0], dtype=np.float32)
#                 is_pad[episode_len:] = 1     
#                 # transform actions into chunkings. "idx" (episode_len, action_horizon):
#                 # [[0, 1, 2...seq-1], [1, 2, 3...seq], [2, 3, 4...seq+1],...]
#                 idx = np.stack([
#                     np.arange(i, i + self.action_horizon)
#                     for i in range(episode_len)
#                 ])
#                 action = action[idx, :]  # (episode_len, action_horizon, action_dim)
#                 is_pad = is_pad[idx]  # (episode_len, action_horizon)
#                 # construct data
#                 images.append(image)
#                 proprios.append(proprio)
#                 actions.append(action)
#                 is_pads.append(is_pad)
#         # combine data from different episodes. "n" depends on
#         # the number of episodes and the horizon of each episode        
#         images = np.concatenate(images, axis=0)  # (n, num_camera, c, h, w)
#         proprios = np.concatenate(proprios, axis=0)  # (n, pos_dim)
#         actions = np.concatenate(actions, axis=0)  # (n, action_horizon, action_dim)
#         is_pads = np.concatenate(is_pads, axis=0)  # (n, action_horizon)
#         # convert np.ndarray to torch.Tensor
#         images = torch.from_numpy(images).float()
#         proprios = torch.from_numpy(proprios).float() 
#         actions = torch.from_numpy(actions).float()
#         is_pads = torch.from_numpy(is_pads).float()
#         # reshape the images
#         image = rearrange(image, 't n h w c -> t n c h w')  # (n, num_camera, c, h, w)
#         # normalize pixel intensity to [0, 1] (if necessary)
#         image = image / 255.0


# def load_data(args):
#     # Construct dataset and dataloader
#     dataset = ACTDataset(args)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=4,
#         prefetch_factor=1
#     )
#     return dataloader, dataset.norm_dict


class ACTDataset(Dataset):
    def __init__(self, args, norm_stats):
        super().__init__()
        self.num_queries = args.action_horizon
        self.dataset_dir = args.dataset_dir
        self.camera_names = args.cameras
        self.norm_stats = norm_stats
        self.full_episode = args.full_episode
        file_paths = os.path.join(self.dataset_dir, '*.h5')
        self.file_paths = glob(file_paths)


    def __len__(self):
        return len(self.file_paths)
    
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with h5py.File(path, 'r') as f:
            action = f['/action'][:]  # (time_steps, action_dim)
            time_steps = action.shape[0]
            if self.full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(time_steps)
            qpos = f['/observations/qpos'][:][start_ts]  # (pos_dim,)
            images = []
            for cam_name in self.camera_names:
                images.append(f[f'/observations/images/{cam_name}'][:][start_ts])        
            # concatenate images
            image = np.stack(images, axis=0)  # (num_camera, h, w, c)
            # normalize actions and joint positions
            action = ((action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]).squeeze()
            qpos = ((qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]).squeeze()
            # action sequence zero padding
            zero_pad = np.zeros((self.num_queries - 1, action.shape[1]), dtype=np.float32)
            action = np.concatenate([action, zero_pad], axis=0)
            is_pad = np.zeros(action.shape[0])
            is_pad[time_steps: ] = 1  # define where sequences of zero padding are
            # get action chunkings
            action_seq = action[start_ts: (start_ts + self.num_queries)]
            is_pad = is_pad[start_ts: (start_ts + self.num_queries)]
            # transform nd.array to torch.tensor
            image = torch.from_numpy(image).permute(0, 3, 1, 2)  # (num_camera, c, h, w)
            qpos = torch.from_numpy(qpos).float()  # (pos_dim,)
            action_seq = torch.from_numpy(action_seq).float()  # (num_queries, action_dim)
            is_pad = torch.from_numpy(is_pad).bool()  # (num_queries,)
            # normalize images pixel intensity to [0, 1] (if necessary)
            image = image / 255.0
            return image, qpos, action_seq, is_pad
                        

def load_data(args):
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(args)
    # Construct dataset and dataloader
    dataset = ACTDataset(args, norm_stats)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=1
    )
    return dataloader, norm_stats
