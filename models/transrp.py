"""
Taken from https://github.com/baoqiangmaUMCG/TransRP
"""


# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Union, List

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

import monai

__all__ = ["TransRP", "TransRP_m1", "TransRP_m2", "TransRP_m3" ]

class TransRP(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels_trf: int = 512,
        img_size: Union[Sequence[int], int] = (6,6,6),
        patch_size: Union[Sequence[int], int] = (1,1,1),
        n_input_channels: int = 3,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_target_classes: List[int] = [1],
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        num_tabular_features: int = 27,
        version: str = "m1",
        **kwargs
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.
            qkv_bias: apply bias to the qkv linear layer in self attention block

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels_trf,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            
        self.num_target_classes = num_target_classes
        self.freeze_encoder = False

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU()
        )
        self.fc_head = nn.Sequential(
            nn.Linear(32, num_target_classes[0]),
            nn.Flatten(start_dim=0) if num_target_classes[0] == 1 else nn.Identity()
        )

        self.cnn = nn.Sequential( *list((monai.networks.nets.resnet18(spatial_dims=3, n_input_channels = n_input_channels, num_classes=  1, conv1_t_stride = 2)).children())[0:8])
        (self.cnn)[7][0].conv1.stride = (1, 1, 1) # set outout feature map with size of 6x6x6x512
        (self.cnn)[7][0].downsample[0].stride = (1, 1, 1)     
        

        self.use_tabular = num_tabular_features > 0
        self.num_tabular_features = num_tabular_features
        self.img_size  = img_size

        assert version in ["m1", "m2", "m3"]
        self.version = version
        match version:
            case "m1":
                self.linear_clc = nn.Sequential(
                    nn.Linear(num_tabular_features, hidden_size), # new
                    nn.ReLU()
                )
            case "m2":
                self.patch_num  = int((img_size[0] / patch_size[0] )* (img_size[1] / patch_size[1]) * (img_size[2] / patch_size[2]))
            case "m3":
                self.fc2 = nn.Sequential(
                    nn.Linear(128 + num_tabular_features, 32),
                    nn.LeakyReLU()
                )


        #print (self.cnn)

    def forward(self, x, x_clc=None):
        #print (x.size())
        x = self.cnn(x)
        #print (x.size())
        x = self.patch_embedding(x)

        if self.use_tabular:
            match self.version:
                case "m1":
                    x_clc = self.linear_clc(x_clc)
                    x_clc = x_clc[:, None, :]
                    x = torch.cat((x_clc, x), dim=1)
                case "m2":
                    x_clc = x_clc[:, None, :]
                    x_clc = x_clc.repeat(1 , self.patch_num, 1 )
                    x[:,:,-self.num_tabular_features:] = x_clc
                case _:
                    pass

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        
        x = self.fc1(torch.mean(x, 1))

        if self.use_tabular and self.version == "m3":
            x = torch.cat([x, x_clc], dim=1)
        x = self.fc2(x)
        x = self.fc_head(x)

        return x , hidden_states_out    

class TransRP_m1(TransRP):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """
    
    def __init__(
        self,
        in_channels_trf: int = 512,
        img_size: Union[Sequence[int], int] = (6,6,6),
        patch_size: Union[Sequence[int], int] = (1,1,1),
        n_input_channels: int = 3,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_target_classes: List[int] = [1],
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        add_clc: bool = False,
        num_tabular_features: int = 27,
        **kwargs
    ) -> None:
        """
        """
        super().__init__(
            in_channels_trf=in_channels_trf,
            img_size=img_size,
            patch_size=patch_size,
            n_input_channels=n_input_channels,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=classification,
            num_target_classes=num_target_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=post_activation,
            qkv_bias=qkv_bias,
            **kwargs            
        )

        self.classification = classification
        self.add_clc  = add_clc
    
        # print (self.cnn)
        self.use_tabular = num_tabular_features > 0
        self.img_size  = img_size
        self.linear_clc = nn.Linear(num_tabular_features, hidden_size) # new
        self.relu_clc = nn.ReLU(inplace=True)

    def forward(self, x, x_clc=None):

        x = self.cnn(x)
        
        '''
        if self.add_clc: # concat clc (5x5x5x7) to patch (5x5x5x512) 
            x_clc = self.linear_clc(x_clc) # project clinical to 64 features, for non_linear
            x_clc = x_clc[:,:, None, None, None]
            x_clc = x_clc.repeat(1 ,1, self.img_size[0],self.img_size[1],self.img_size[2] )
            x = torch.cat((x ,x_clc ), 1)
        '''
        x = self.patch_embedding(x)
        if self.use_tabular:         
            x_clc  = self.linear_clc(x_clc)
            x_clc  = self.relu_clc(x_clc) # non_linear
            x_clc = x_clc[:, None, :]
            x = torch.cat((x_clc, x), dim=1)
        
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        '''
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        '''
        x = self.fc1(torch.mean(x, 1))
        x = self.fc2(x)
        x = self.fc_head(x)

        return x , hidden_states_out    

class TransRP_m2(TransRP):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """
    def __init__(
        self,
        in_channels_trf: int = 512,
        img_size: Union[Sequence[int], int] = (6,6,6),
        patch_size: Union[Sequence[int], int] = (1,1,1),
        n_input_channels: int = 3,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_target_classes: List[int] = [1],
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        add_clc: bool = False,
        num_tabular_features: int = 27,
        **kwargs
    ) -> None:
        """
        """
        super().__init__(
            in_channels_trf=in_channels_trf,
            img_size=img_size,
            patch_size=patch_size,
            n_input_channels=n_input_channels,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=classification,
            num_target_classes=num_target_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=post_activation,
            qkv_bias=qkv_bias,
            **kwargs            
        )
        #print (self.cnn)
        self.use_tabular = num_tabular_features > 0
        self.num_tabular_features = num_tabular_features
        self.img_size  = img_size
        self.patch_num  = int((img_size[0] / patch_size[0] )* (img_size[1] / patch_size[1]) * (img_size[2] / patch_size[2]))

    def forward(self, x, x_clc):

        x = self.cnn(x)    
        x = self.patch_embedding(x)
        if self.use_tabular:          
            x_clc = x_clc[:, None, :]
            x_clc = x_clc.repeat(1 , self.patch_num, 1 )
            #x = torch.cat((x_clc, x), dim=2)
            #print (x.size())
            x[:,:,-self.num_tabular_features:] = x_clc
            
            #print (x.size())
        
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        x = self.fc1(torch.mean(x, 1))
        x = self.fc2(x)
        x = self.fc_head(x)

        return x, hidden_states_out        
    
class TransRP_m3(TransRP):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels_trf: int = 512,
        img_size: Union[Sequence[int], int] = (6,6,6),
        patch_size: Union[Sequence[int], int] = (1,1,1),
        n_input_channels: int = 3,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_target_classes: List[int] = [1],
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        add_clc: bool = False,
        num_tabular_features: int = 27,
        **kwargs
    ) -> None:
        """
        """
        super().__init__(
            in_channels_trf=in_channels_trf,
            img_size=img_size,
            patch_size=patch_size,
            n_input_channels=n_input_channels,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=classification,
            num_target_classes=num_target_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=post_activation,
            qkv_bias=qkv_bias,
            **kwargs            
        )

        self.img_size  = img_size
        self.use_tabular = num_tabular_features > 0

        self.fc2 = nn.Sequential(
            nn.Linear(128 + num_tabular_features, 32),
            nn.LeakyReLU()
        )

    def forward(self, x, x_clc):

        x = self.cnn(x)
        '''
        if self.add_clc: # concat clc (5x5x5x7) to patch (5x5x5x512) 
            x_clc = self.linear_clc(x_clc) # project clinical to 64 features, for non_linear
            x_clc = x_clc[:,:, None, None, None]
            x_clc = x_clc.repeat(1 ,1, self.img_size[0],self.img_size[1],self.img_size[2] )
            x = torch.cat((x ,x_clc ), 1)
        '''
        x = self.patch_embedding(x)
        '''
        if self.add_clc: # concat clc (5x5x5x7) to patch (5x5x5x512) 
            
            x_clc  = self.linear_clc(x_clc)
            x_clc = x_clc[:, None, :]
            x = torch.cat((x_clc, x), dim=1)
        '''
        
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        x = self.fc1(torch.mean(x, 1))
        if self.use_tabular:
            x = torch.cat([x, x_clc], dim=1)
        x = self.fc2(x)
        x = self.fc_head(x)

        return x, hidden_states_out
    
