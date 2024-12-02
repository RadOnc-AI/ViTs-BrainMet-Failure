from typing import Type, Union, List
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes,track_running_stats=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes,track_running_stats=False)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes,track_running_stats=False)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    
    block_inplanes=[64, 128, 256, 512]
    block: Union[
        Type[BasicBlock],
        Type[Bottleneck]
    ]
    layers: tuple[int,int,int,int]

    def __init__(
            self,
            block_inplanes=[64, 128, 256, 512],
            n_input_channels=3,
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            shortcut_type='B',
            widen_factor=1.0,
            num_tabular_features=27,
            num_target_classes: List[int] = [1],
            pretrain_path: str = None,
            **kwargs):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, self.block_inplanes[0], self.layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(self.block,
                                       self.block_inplanes[1],
                                       self.layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(self.block,
                                       self.block_inplanes[2],
                                       self.layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(self.block,
                                       self.block_inplanes[3],
                                       self.layers[3],
                                       shortcut_type,
                                       stride=2)
        
        # Load pretrained backbone
        ## todo: add option to load only part of the model & check compatibility 
        if pretrain_path is not None:
            self._load_pretrained(pretrain_path)

    
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(self.block_inplanes[3] * self.block.expansion, 128 * self.block.expansion),
            nn.LeakyReLU()
            )
        
        self.use_tabular = num_tabular_features > 0
        self.fc2 = nn.Sequential(
            nn.Linear(128 * self.block.expansion + num_tabular_features, 32* self.block.expansion),
            nn.LeakyReLU()
            )

        self.num_target_classes = num_target_classes
        # if not isinstance(self.num_target_classes, list):
        #     self.num_target_classes = [self.num_target_classes]
        
        self.fc_head = nn.Sequential(
            nn.Linear(32 * self.block.expansion, self.num_target_classes[0]),
            nn.Flatten(start_dim=0) if self.num_target_classes[0] == 1 else nn.Identity()
        )

        #for compatibility with TrainingBase
        
        self.freeze_encoder = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm3d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
    def _load_pretrained(self, model_path,ignore_layers: List[str] = []):
        state_dict = torch.load(model_path,map_location="cpu")["state_dict"]
        state_dict = {k[6:]: v for k, v in state_dict.items() if k[6:] not in ignore_layers}
        self.load_state_dict(state_dict)
    
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion,track_running_stats=False))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x, tabular):
        x = self.encode(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        if self.use_tabular:
            x = torch.cat([x,tabular],dim=1)
        x = self.fc2(x)

        x = self.fc_head(x)
        
        return [x]


class ResNet10(ResNet):

    layers = (1, 1, 1, 1)
    block = BasicBlock

class ResNet18(ResNet):

    layers = (2, 2, 2, 2)
    block = BasicBlock

class ResNet34(ResNet):

    layers = (3, 4, 6, 3)
    block = BasicBlock

class ResNet50(ResNet):

    layers = (3, 4, 6, 3)
    block = Bottleneck

class ResNet101(ResNet):
    
    layers = (3, 4, 23, 3)
    block = Bottleneck

class ResNet152(ResNet):
    
    layers = (3, 8, 36, 3)
    block = Bottleneck

class ResNet200(ResNet):

    layers = (3, 24, 36, 3)
    block = Bottleneck