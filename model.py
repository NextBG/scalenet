'''
ScaleNet model for predicting scale factor between two images
'''

import torch
import torch.nn as nn

import math
from efficientnet_pytorch import EfficientNet
from utils import replace_bn_with_gn

class ScaleNet(nn.Module):
    def __init__(
            self,
            enc_dim: int,
            sa_layers: int,
            sa_heads: int,
            sa_ff_dim_factor: int,
        ):
        super().__init__()
        self.enc_dim = enc_dim

        # Observation Encoder
        self.obs_encoder = EfficientNet.from_name('efficientnet-b0', in_channels=6, include_top=False)
        replace_bn_with_gn(self.obs_encoder)

        # Transformation Encoder
        self.transform_encoder = nn.Linear(3, self.enc_dim)

        # Compression layer for observation encoder
        self.compress_obs_enc = nn.Linear(1280, self.enc_dim) # B0: 1280, B1: 1536, B2: 2304, B3: 3072, B4: 3840, B5: 4864, B6: 6144, B7: 8192

        # Self-attention encoder
        self.positional_encoding = PositionalEncoding(self.enc_dim)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.enc_dim,
            nhead=sa_heads,
            dim_feedforward=sa_ff_dim_factor*self.enc_dim,
            activation='gelu',
            batch_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=sa_layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.enc_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs_0: torch.tensor, obs_1:torch.tensor, transform: torch.tensor):
        # Encode observation
        BS = obs_0.shape[0]

        # Encode transformation
        trans_enc = self.transform_encoder(transform)
        trans_enc = trans_enc.unsqueeze(1)  #[B, 1, E]

        # 1) Fusion and mean pool
        obs = torch.cat([obs_0, obs_1], dim=1)                            # [B, 3, H, W] -> [B, 6, H, W]
        obs_enc = self.obs_encoder.extract_features(obs)                  # [B, 1280, 4, 8]
        obs_enc = self.obs_encoder._avg_pooling(obs_enc)                  # [B, 1280, 1, 1]
        obs_enc = obs_enc.view(BS, -1)                                    # [B, 1280]
        obs_enc = self.compress_obs_enc(obs_enc)                          # [B, E]
        obs_enc = obs_enc.unsqueeze(1)                                    # [B, 1, E]
        context_enc = torch.cat([obs_enc, trans_enc], dim=1)              # [B, 2, E]

        # 2) Fusion and flatten
        # obs = torch.cat([obs_0, obs_1], dim=1)                              # [B, 3, H, W] -> [B, 6, H, W]
        # obs_enc = self.obs_encoder.extract_features(obs)                    # [B, 1280, 4, 8]
        # obs_enc = obs_enc.view(BS, 1280, -1)                                # [B, 1280, 32]
        # obs_enc = obs_enc.permute(0, 2, 1)                                  # [B, 32, 1280]
        # obs_enc = self.compress_obs_enc(obs_enc)                            # [B, 32, E]
        # context_enc = torch.cat([obs_enc, trans_enc], dim=1)                # [B, 33, E]

        # 3) No-fusion and flatten
        # obs = torch.cat([obs_0, obs_1], dim=0)                            # [B, 3, H, W] -> [B*2, 3, H, W]
        # obs_enc = self.obs_encoder.extract_features(obs)                  # [B*2, 3, H, W] -> [B*2, 1280, 4, 8]
        # obs_enc = obs_enc.view(BS*2, 1280, -1)                            # [B*2, 1280, 32]
        # obs_enc = obs_enc.permute(0, 2, 1)                                # [B*2, 32, 1280]
        # obs_enc = self.compress_obs_enc(obs_enc)                          # [B*2, 32, E]
        # obs_0_enc, obs_1_enc = torch.split(obs_enc, BS, dim=0)            # [B, 32, E], [B, 32, E]
        # context_enc = torch.cat([obs_0_enc, obs_1_enc, trans_enc], dim=1) # [B, 65, E]

        # Self-attention
        context_enc = self.positional_encoding(context_enc)
        context_enc = self.sa_encoder(context_enc)
        
        # Mean pool
        context_enc = torch.mean(context_enc, dim=1) #[B, E]

        # Decode
        scale_factor = self.decoder(context_enc) #[B, 1]

        return scale_factor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=2):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x