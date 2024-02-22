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
        self.kl_weight = args['kl_weight']
        

    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        ###============ ??? ============###
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        ###============ ??? ============###
        
        ### Training
        if actions is not None:
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, None, actions, is_pad, vq_sample)

            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            
            return loss_dict

        ### Inference
        else:
            a_hat, _, (_, _), _, _ = self.model(qpos, image, None, vq_sample=vq_sample) # no action, sample from prior
            
            return a_hat

    
    def configure_optimizers(self):
        return self.optimizer
