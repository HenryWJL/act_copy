import os
import sys
import argparse
import tomli
import torch
from tqdm import tqdm

from utils import load_data
from policy import ACTPolicy

import IPython
e = IPython.embed


def make_parser():
    parser = argparse.ArgumentParser(
        description="Train the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        default="",
        help="Directory used for loading training data."
    )
    parser.add_argument(
        "--ckpt_dir",
        default="",
        help="Directory used for saving models."
    )
    
    return parser


def train(args):
    device = torch.cuda(args.device)
    
    train_dataloader = load_data(args)
    policy = ACTPolicy(args).to(device)
    optimizer = policy.configure_optimizers().to(device)
    
    for epoch in tqdm(range(args.epoch)):
        policy.train()
        optimizer.zero_grad()
        for batch_idx, (image, qpos, action) in enumerate(train_dataloader):
            image, qpos, action = image.to(device), qpos.to(device), action.to(device)
            loss = policy(qpos, image, action)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 100 == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'epoch_{epoch + 1}_seed_{args.seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            

def main(argv=sys.argv[1:]):
    
    parser = make_parser()
    args = parser.parse_args(argv)

    with open('config.toml', 'rb') as f:
        _config = tomli.load(f)
        
        args.backbone = str(_config['model']['backbone'])
        args.lr_backbone = float(_config['model']['lr_backbone'])
        args.num_queries = int(_config['model']['num_queries'])
        args.hidden_dim = int(_config['model']['hidden_dim'])
        args.nheads = int(_config['model']['nheads'])
        args.dim_feedforward = int(_config['model']['dim_feedforward'])
        args.enc_layers = int(_config['model']['enc_layers'])
        args.enc_layers = int(_config['model']['enc_layers'])
        
        args.kl_weight = float(_config['train']['kl_weight'])
        args.lr = float(_config['train']['lr'])
        args.batch = int(_config['train']['batch'])
        args.epoch = int(_config['train']['epoch'])
        args.seed = int(_config['train']['seed'])
        args.device = str(_config['train']['device'])
        
    train(args)
