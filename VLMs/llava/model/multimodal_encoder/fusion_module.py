
import torch
import torch.nn as nn
from .ctvit import CTViT

class PETCTFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.pet_enc = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )

        pet_ckpt = '/home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/pet_emb/ctvit.76000.pt'
        checkpoint = torch.load(pet_ckpt, map_location='cpu')
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
        self.pet_enc.load_state_dict(checkpoint) 

        for param in self.pet_enc.parameters():
            param.requires_grad = False

        self.pet_enc.eval() 

        self.ct_enc = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )

        ct_ckpt = '/home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/ct_emb/ctvit.76000.pt'
        checkpoint = torch.load(ct_ckpt, map_location='cpu')
        from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
        consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
        self.ct_enc.load_state_dict(checkpoint) 

        for param in self.ct_enc.parameters():
            param.requires_grad = False

        self.ct_enc.eval() 

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Cross-Attention module
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Important for (B, N, C)
        )

        # Optional: LayerNorm before and after attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed Forward after cross-attn
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, pet_img, ct_img):
        with torch.no_grad(): 
            pet_emb = self.pet_enc(pet_img)
            ct_emb = self.ct_enc(ct_img)
        
        B, D, H, W, C = pet_emb.shape

        # Reshape (B, D, H, W, C) --> (B, N, C)
        pet_tokens = pet_emb.contiguous().view(B, D * H * W, C)
        ct_tokens  = ct_emb.contiguous().view(B, D * H * W, C)

        # Normalize inputs before attention
        pet_norm = self.norm1(pet_tokens)
        ct_norm  = self.norm1(ct_tokens)

        # Cross Attention: Query = PET, Key/Value = CT
        attn_output, _ = self.cross_attn(query=pet_norm, key=ct_norm, value=ct_norm)

        # Skip connection from PET tokens
        fused = pet_tokens + attn_output

        # Feed Forward + second skip connection
        fused_norm = self.norm2(fused)
        fused = fused + self.ffn(fused_norm)
        
        fused = fused.contiguous().view(B, D, H, W, C)
        
        return fused 