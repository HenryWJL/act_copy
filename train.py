import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
import argparse
import tomli
import torch
from tqdm import tqdm
from utils import Logger, load_data, set_seed
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
        type=str,
        default="./data/cup_place",
        help="Directory used for loading training data."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Directory used for loading pretrained models."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./experiments",
        help="Directory used for saving models."
    )
    return parser


def train_one_epoch(dataloader, policy, optimizer, device):
    total_loss = 0.0
    for _, (image, qpos, action, is_pad) in enumerate(dataloader):
        optimizer.zero_grad()
        image, qpos, action, is_pad = image.to(device), \
            qpos.to(device), action.to(device), is_pad.to(device)
        loss = policy(qpos, image, action, is_pad)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss = total_loss / len(dataloader)
    return loss


def train(args):
    torch.cuda.empty_cache()
    # get device
    torch.cuda.set_device(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    # load checkpoints (if any)
    ckpt = None
    if args.checkpoint is not None:
        dataset_dir = args.dataset_dir
        save_dir = args.save_dir
        epoch = args.epoch
        ckpt = torch.load(args.checkpoint, map_location=device)
        args = ckpt["args"]
        args.dataset_dir = dataset_dir
        args.save_dir = save_dir
        args.epoch = epoch
    # create saving directory
    save_dir = os.path.join(
        args.save_dir,
        f'seed_{args.seed}_horizon_{args.action_horizon}_lr_{args.lr}_kl_{args.kl_weight}'
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, "checkpoints"))
    # create logger
    log_path = os.path.join(save_dir, "log.txt")
    logger = Logger(log_path)
    # set seed
    set_seed(args.seed)
    logger.dump(f"seed: {args.seed}, device: {device}")
    # load data
    logger.dump("Loading Data...")
    train_dataloader, _ = load_data(args)
    # instantiate policy and optimizer
    logger.dump("Getting Policy...")
    policy = ACTPolicy(args).to(device)
    optimizer = policy.configure_optimizers()
    if ckpt is not None:
        policy.model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    logger.dump(f"Number of parameters: {policy.model.__repr__()}")
    # train
    logger.dump("Training...")
    policy.train()
    start_epoch = (ckpt["epoch"] + 1) if ckpt is not None else 0
    assert start_epoch < args.epoch
    for epoch in tqdm(range(start_epoch, args.epoch)):
        loss = train_one_epoch(train_dataloader, policy, optimizer, device)
        logger.dump(f"In epoch[{epoch + 1}, {args.epoch}], the loss is: {loss}")
        if (epoch + 1) % args.save_epochs == 0:
            save_path = os.path.join(save_dir, "checkpoints", f'epoch_{epoch + 1}.pth')
            torch.save(
                {
                    "args": args,
                    "epoch": epoch,
                    "norm_stats": train_dataloader.dataset.norm_stats,
                    "model": policy.model.state_dict(),
                    "optimizer": optimizer.state_dict()
                },
                save_path
            )
            

def main(argv=sys.argv[1:]):
    parser = make_parser()
    args = parser.parse_args(argv)
    with open('config.toml', 'rb') as f:
        _config = tomli.load(f)
        # dataset
        args.cameras = list(_config['dataset']['cameras'])
        args.full_episode = bool(_config['dataset']['full_episode'])
        args.norm_mode = str(_config['dataset']['norm_mode'])
        # model
        args.backbone = str(_config['model']['backbone'])
        args.lr_backbone = float(_config['model']['lr_backbone'])
        args.no_encoder = bool(_config['model']['no_encoder'])
        args.state_dim = int(_config['model']['state_dim'])
        args.action_dim = int(_config['model']['action_dim'])
        args.action_horizon = int(_config['model']['action_horizon'])
        args.latent_dim = int(_config['model']['latent_dim'])
        args.hidden_dim = int(_config['model']['hidden_dim'])
        args.nheads = int(_config['model']['nheads'])
        args.dim_feedforward = int(_config['model']['dim_feedforward'])
        args.enc_layers = int(_config['model']['enc_layers'])
        args.dec_layers = int(_config['model']['enc_layers'])
        args.dropout = float(_config['model']['dropout'])
        args.pre_norm = bool(_config['model']['pre_norm'])
        # train
        args.kl_weight = float(_config['train']['kl_weight'])
        args.lr = float(_config['train']['lr'])
        args.weight_decay = float(_config['train']['weight_decay'])
        args.batch = int(_config['train']['batch'])
        args.epoch = int(_config['train']['epoch'])
        args.seed = int(_config['train']['seed'])
        args.save_epochs = int(_config['train']['save_epochs'])
    train(args)


if __name__ == '__main__':
    main()