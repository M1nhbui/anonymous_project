import torch
import torch.nn as nn
import re
from .helpers import PerceiverResampler

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    
    
class Resampler(nn.Module):
    def __init__(self, embedding_dim=4096, vis_dim=512, perceiver_num=64):
        super().__init__()
        self.perceiver = PerceiverResampler(dim=vis_dim, num_latents=perceiver_num)
        self.fc = nn.Linear(vis_dim, embedding_dim)
        
    def forward(self, x, return_attn=False):
        B, D, H, W, C = x.shape
        x = x.view(B, 1, 1, D * H * W, C)
        
        if return_attn:
            x, attn = self.perceiver(x, return_attn)
        else:
            x = self.perceiver(x, return_attn)
            
        x = x.view(B, -1, x.shape[3])
        x = self.fc(x)
        
        if return_attn:
            return x, attn
        return x 

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(projector_type, config.mm_hidden_size, config.hidden_size)
    if projector_type == 'linear':
        return Resampler(embedding_dim=config.hidden_size, vis_dim=config.mm_hidden_size, perceiver_num=64)
        # return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
