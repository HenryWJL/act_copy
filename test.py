import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
import argparse
import numpy as np
import torch
from policy import ACTPolicy
from utils import set_seed

import IPython
e = IPython.embed

def make_parser():
    parser = argparse.ArgumentParser(
        description="Train the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./experiment/seed_52_horizon_10_lr_0.0005_kl_10.0/checkpoints/epoch_1000.pth",
        help="Checkpoint path."
    )
    return parser


@torch.no_grad()
def test(checkpoint: str, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
    # load checkpoint
    ckpt = torch.load(checkpoint)
    train_args = ckpt["args"]
    # get device
    set_seed(train_args.seed)
    torch.cuda.empty_cache()
    torch.cuda.set_device(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # instantiate policy
    policy = ACTPolicy(train_args).to(device)
    policy.model.load_state_dict(ckpt["model"])
    # get norm status
    norm_stats = ckpt["norm_stats"]
    # inference
    policy.eval()
    image, qpos = image.to(device), qpos.to(device)
    action_pred = policy(qpos, image)
    # unnormalize actions
    action_pred = action_pred * norm_stats["action_std"] + norm_stats["action_mean"]
    return action_pred


def main(argv=sys.argv[1:]):
    parser = make_parser()
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint
    image = torch.rand(1, 1, 3, 480, 640)
    qpos = torch.rand(1, 7)
    action_pred = test(checkpoint, image, qpos)
    print(action_pred)
    
    
if __name__ == '__main__':
    main()


# a_hat = None
# norm_dict = None
# # unnormalize the actions
# if "max" in norm_dict["action"].keys():
#     a_max = norm_dict["action"]["max"]
#     a_max = torch.from_numpy(a_max).float().detach()[None,...]
#     a_hat = a_hat * a_max
# elif "mean" in norm_dict["action"].keys():
#     a_mean = norm_dict["action"]["mean"]
#     a_std = norm_dict["action"]["std"]
#     a_mean = torch.from_numpy(a_mean).float().detach()[None,...]
#     a_std = torch.from_numpy(a_std).float().detach()[None,...]
#     a_hat = a_hat * a_std + a_mean


