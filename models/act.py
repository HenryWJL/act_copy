import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps

def get_sinusoid_encoding_table(n_position, d_hid):
    """1D sinusoidal positional encoding"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ACT(nn.Module):

    def __init__(self, backbone, transformer, encoder, state_dim, action_dim, num_queries, latent_dim, camera_names):
        """ACT model, a variant of DERT VAE model
        Params:
        
            backbones: visual encoder backbone.
            
            encoder: the encoder of CVAE (a Transformer encoder architecture)
            
            transformer: the decoder of CVAE (a Transformer architecture).
            
            state_dim: dimension of robot state (14: joint positions of two arms) 

            action_dim: dimension of action (14: joint positions of two arms)
            
            num_queries: length of action sequence

            latent dim: dimension of latent z's mu and logvar

            camera_names: a list of camera names (str)
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        # VAE encoder
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra [CLS] token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action sequence to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project joint positions to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project [CLS] output to latent z's mu and logvar
        """
        Get 1D sinusoid position embedding for encoder's inputs ([CLS] + joint_pos + action_seq)
        
        Using "register_buffer", 'pos_table' will not be considered as a parameter of the model, i.e., its value will 
        be fixed during back propagation
        """
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], joint_pos, action_seq
        # VAE decoder
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)  # project joint positions to proprio embedding
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent z to latent embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # additional learned position embedding for proprio and latent embeddings
        self.backbones = nn.ModuleList([backbone for i in range(len(camera_names))])
        self.input_proj = nn.Conv2d(self.backbones[0].num_channels, hidden_dim, kernel_size=1)  # project backbone's image features to embedding
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # learned position embedding of Transformer decoder's query
        
    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        Params:
            qpos: joint positions (batch, 14)
            
            image: image observations (batch, num_cam, channel, height, width)
            
            actions: action sequences (batch, seq, action_dim)

            is_pad: a bool vector indicating which part of the action sequence is zero padded
        """
        ### VAE encoder
        latent_input, mu, logvar = self.encode(qpos, actions, is_pad)
        ### VAE decoder
        # Image observation features and their position embeddings
        all_cam_features = []
        all_cam_pos = []
        for cam_id,_ in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            # If "return_interm_layers" is set to True, the backbone 
            # will return features from intermediate layers
            features = features[0]  # take the feature from the last layer
            pos = pos[0]  # take the pos from the last layer
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        # proprioception features (joint positions embedding)
        proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        hs = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight
        )[0]
        a_hat = self.action_head(hs)
        return a_hat, (mu, logvar)
    
    def encode(self, qpos, actions=None, is_pad=None):
        """Obtain latent z and project it to embedding"""
        bs, _ = qpos.shape
        ### Inference
        if self.encoder is None:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)  
        ### Training or validation
        else:
            is_training = actions is not None
            ### Training
            if is_training:
                # get input embedding
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, 2+seq, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (2+seq, bs, hidden_dim)
                # get 1D sinusoidal position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (2+seq, 1, hidden_dim)
                # do not mask [CLS] and qpos tokens
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)  # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, 2+seq)
                # query the model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                # get latent z
                encoder_output = encoder_output[0]  # take [CLS] output only
                latent_info = self.latent_proj(encoder_output)  # get z's mu + logvar
                mu = latent_info[:, :self.latent_dim]
                logvar = latent_info[:, self.latent_dim:]
                latent_sample = reparametrize(mu, logvar)
                # get latent z embedding
                latent_input = self.latent_out_proj(latent_sample)
            ### Validation
            else:
                mu = logvar = None
                latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                latent_input = self.latent_out_proj(latent_sample)
        return latent_input, mu, logvar
    
    def __repr__(self):
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_parameters / 1e6

def build_encoder(args):
    """Build VAE encoder"""
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = args.pre_norm
    activation = "relu"
    encoder_layer = TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(
        encoder_layer,
        num_encoder_layers,
        encoder_norm
    )
    return encoder

def build_ACT_model_and_optimizer(args):
    # Build VAE encoder
    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)
    # Build visual encoder backbone  
    backbone = build_backbone(args)    
    # Build VAE decoder
    transformer = build_transformer(args)
    # Build ACT model
    model = ACT(
        backbone=backbone,
        transformer=transformer,
        encoder=encoder,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_queries=args.action_horizon,
        latent_dim = args.latent_dim,
        camera_names=args.cameras
    )
    # Build optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    return model, optimizer
