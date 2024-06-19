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
        default="",
        help="Checkpoint path."
    )
    return parser


@torch.no_grad()
def test(checkpoint: str, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
    torch.cuda.empty_cache()
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    train_args = ckpt["args"]
    # set seed
    set_seed(train_args.seed)
    # instantiate policy
    policy = ACTPolicy(train_args).to(device)
    policy.model.load_state_dict(ckpt["model"])
    # get norm status
    norm_stats = ckpt["norm_stats"]
    # normalize qpos
    qpos = (qpos - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
    # inference
    policy.eval()
    image, qpos = image.to(device), qpos.to(device)
    action_pred = policy(qpos, image)
    action_pred = action_pred.cpu()
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
    
    
if __name__ == '__main__':
    main()
