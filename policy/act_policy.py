import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np

from models import build_ACT_model_and_optimizer

import IPython
e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args.kl_weight
        

    def __call__(self, qpos, image, actions=None, is_pad=None):
        ###============ ??? ============###
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        ###============ ??? ============###
        
        ### Training
        if actions is not None:
            a_hat, (mu, logvar) = self.model(qpos, image, actions, is_pad)

            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss = l1 + total_kld[0] * self.kl_weight
            
            return loss

        ### Inference
        else:
            a_hat, (_, _) = self.model(qpos, image)
            
            return a_hat

    
    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
