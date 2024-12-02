"""
Adapted from https://github.com/hila-chefer/Transformer-MM-Explainability/tree/main to MONAI's ViT implementation
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from monai.networks.layers import get_act_layer
from monai.utils import look_up_option, deprecated_arg, optional_import
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
# from monai.networks.blocks.transformerblock import TransformerBlock

from einops import rearrange

from models.LRP.layers import *

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_DROPOUT_MODE = {"vit", "swin"}


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    
    return joint_attention
    



class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, act: tuple | str = "GELU", dropout_mode="vit"
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: fraction of the input units to drop.
            act: activation type and arguments. Defaults to GELU. Also supports "GEGLU" and others.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = Linear(hidden_size, mlp_dim) if act != "GEGLU" else Linear(hidden_size, mlp_dim * 2)
        self.linear2 = Linear(mlp_dim, hidden_size)
        self.fn = GELU() # get_act_layer(act)
        self.drop1 = Dropout(dropout_rate)
        dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        if dropout_opt == "vit":
            self.drop2 = Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x
    
    def relprop(self, cam, **kwargs):
        cam = self.drop2.relprop(cam, **kwargs)
        cam = self.linear2.relprop(cam, **kwargs)
        cam = self.drop1.relprop(cam, **kwargs)
        cam = self.fn.relprop(cam, **kwargs)
        cam = self.linear1.relprop(cam, **kwargs)
        return cam

class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = Linear(hidden_size, hidden_size)
        self.qkv = Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = Dropout(dropout_rate)
        self.drop_weights = Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

        self.matmul1 = einsum("blxd,blyd->blxy")
        self.softmax = Softmax(dim=-1)  
        self.matmul2 = einsum("bhxy,bhyd->bhxd")

        self.attn_cam = None
        self.att_mat = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.att_mat

    def save_attn_mat(self, attn):
        self.att_mat = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = self.matmul1([q,k]) * self.scale
        att_mat = self.softmax(att_mat)

        self.save_v(v)

        # if self.save_attn:
        #     # no gradients and new tensor;
        #     # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
        #     self.att_mat = att_mat.detach()
        self.save_attn_mat(att_mat)
        att_mat.register_hook(self.save_attn_gradients)

        att_mat = self.drop_weights(att_mat)
        x = self.matmul2([att_mat, v])
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x
    
    def relprop(self, cam, **kwargs):
        cam = self.drop_output.relprop(cam, **kwargs)
        cam = self.out_proj.relprop(cam, **kwargs)
        cam = rearrange(cam, "b l (h d) -> b h l d", h=self.num_heads)
        
        cam1, cam_v = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.drop_weights.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        cam_q, cam_k = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)
        


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2 = LayerNorm(hidden_size)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x,2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x,2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x
    
    def relprop(self, cam, **kwargs):
        cam1, cam2 = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        cam1, cam2 = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class ViT_LRP(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
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
        num_register_tokens=4,
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
        **kwargs
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            proj_type (str, optional): patch embedding layer type. Defaults to "conv".
            pos_embed_type (str, optional): position embedding type. Defaults to "learnable".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), proj_type='conv', pos_embed_type='sincos')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), proj_type='conv', pos_embed_type='sincos', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), proj_type='conv', pos_embed_type='sincos', classification=True,
            >>>           spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = True
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=n_input_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(hidden_size)
        # if self.classification:
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore
        self.classification_head = nn.Identity()

        # self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, hidden_size))

        self.add = Add()
        self.pool = IndexSelect()
        self.cat = Cat()

        self.inp_grad = None

        self.num_target_classes = num_target_classes
        self.use_tabular = num_tabular_features > 0
        self.freeze_encoder = False

        self.fc1 = Sequential(
            Linear(hidden_size, 128),
            LeakyReLU()
        )
        self.fc2 = Sequential(
            Linear(128 + num_tabular_features, 32),
            LeakyReLU()
        )
        self.fc_head = Sequential(
            Linear(32, num_target_classes[0]),
            )
        
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, hidden_size))


    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad


    def forward(self, x, tabular=None):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        if hasattr(self, "register_tokens"):
            r = self.register_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, r), dim=1)

        x.register_hook(self.save_inp_grad)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        x = self.pool(x,dim=1,indices=torch.tensor(0,device=x.device)).squeeze(1)

        x = self.fc1(x)
        if self.use_tabular:
            x = self.cat([x, tabular], dim=1)
        x = self.fc2(x)
        x = self.fc_head(x)
        x = torch.flatten(x, start_dim=0) if self.num_target_classes[0] == 1 else x
        # x = self.classification_head(x[:, 0])
        return x, hidden_states_out
    
    def relprop(self, cam, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # cam = cam.unsqueeze(1)
        cam = self.fc_head.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam, _ = self.cat.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        if method == "full":
            raise NotImplementedError("Conv3d LRP not implemented")
        elif method == "rollout":
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = attn_heads.mean(dim=1).detach()
                if hasattr(self,"register_tokens"):
                    avg_heads = avg_heads[:, :-self.register_tokens.shape[0], :-self.register_tokens.shape[0]]
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            # if hasattr(self,"register_tokens"):
            #     return cam[:,0, 1:-self.register_tokens.shape[0]]
            # else:
            return cam[:,0, 1:]
            
        elif method == "transformer_attribution":
            cams = []        
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                if hasattr(self,"register_tokens"):
                    grad = grad[:,:, :-self.register_tokens.shape[0], :-self.register_tokens.shape[0]]
                    cam = cam[:,:, :-self.register_tokens.shape[0], :-self.register_tokens.shape[0]]
                # cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                # grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=1)
                cams.append(cam)
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            
            # if hasattr(self,"register_tokens"):
            #     return rollout[:,0, 1:-self.register_tokens.shape[0]]
            # else:
            return rollout[:,0, 1:]
            
