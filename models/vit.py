from typing import Type, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT 



class ViT(ViT):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        # num_classes: int,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        # channels: int = 3,
        spatial_dims: int = 3,
        emb_dropout: float = 0.1,
        dropout_rate: float = 0.1,
        n_input_channels=3,
        num_tabular_features=27,
        num_target_classes=[1],
        num_register_tokens=0,
        save_attn: bool = False,
        **kwargs
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            spatial_dims=spatial_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            in_channels=n_input_channels,
            classification=True,
            save_attn=save_attn,
        )
        self.classification_head = nn.Identity()
        # self.conv1 = nn.Conv2d(channels, dim, patch_size, patch_size)
        # self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(dropout)
        self.num_target_classes = num_target_classes
        self.use_tabular = num_tabular_features > 0
        self.freeze_encoder = False

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128 + num_tabular_features, 32),
            nn.LeakyReLU()
        )
        self.fc_head = nn.Sequential(
            nn.Linear(32, num_target_classes[0]),
            nn.Flatten(start_dim=0) if num_target_classes[0] == 1 else nn.Identity()
        )
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, hidden_size))

    def forward(self, x, tabular=None):
        # x, hidden_states_out = super().forward(x)

        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        if hasattr(self, "register_tokens"):
            r = self.register_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, r), dim=1)
            
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])

        x = self.fc1(x)
        if self.use_tabular:
            x = torch.cat([x, tabular], dim=1)
        x = self.fc2(x)
        x = self.fc_head(x)

        return x, hidden_states_out