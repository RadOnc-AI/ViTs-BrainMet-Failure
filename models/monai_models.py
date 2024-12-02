from torch import nn, load, no_grad, Tensor, cat, zeros_like
import monai.networks.nets as nets
from pathlib import Path
from copy import deepcopy
from typing import Union, List


class MonaiNets(nn.Module):
    """
    Wrapper around MonAI architectures to use them as image encoders & combine them with clinical features

    Arguments:
        backbone:   dict, as if you are calling the original MonAI network with its name and arguments
        remove_last_backbone_layer: bool, you can replace the last linear layer of backbone with nn.Identity or you can use it

    """

    def __init__(
        self,
        backbone: dict = {"resnet50": {}},
        remove_last_backbone_layer: bool = False,
        pretrained_path: str = None,
        num_tabular_features: int = 0,
        num_target_classes: Union[int, List[int]] = 1,
        freeze_encoder_until: int = None,
    ) -> None:
        super().__init__()
        backbone_name = list(backbone.keys())[0]
        backbone_args = backbone[backbone_name]
        self.backbone = getattr(nets, backbone_name)(**backbone_args)
        if remove_last_backbone_layer:
            if "resnet" in backbone_name:
                out_size = self.backbone.fc.weight.shape[1]
                self.backbone.fc = nn.Identity()
            elif "densenet" in backbone_name.lower():
                out_size = self.backbone.class_layers.out.shape[1]
                self.backbone.class_layers.out = nn.Identity()
            elif "senet" in backbone_name.lower():
                out_size = self.backbone.last_linear.shape[1]
                self.backbone.last_linear = nn.Identity()
        else:
            if "resnet" in backbone_name:
                out_size = self.backbone.fc.weight.shape[0]
            elif "densenet" in backbone_name.lower():
                out_size = self.backbone.class_layers.out.shape[0]
            elif "senet" in backbone_name.lower():
                out_size = self.backbone.last_linear.shape[0]

        
        if pretrained_path:
            assert "resnet" in backbone_name, "other  models not implemented"
            # assert backbone_name in pretrained_path, "wrong model and weight selection"
            print("Loading pretrained weights from: ", pretrained_path)
            pretrained_path = Path(pretrained_path).expanduser().resolve()
            state_dict = load(pretrained_path, map_location="cpu")["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            in_channels = self.backbone.conv1.weight.shape[1]
            state_dict["conv1.weight"] = (
                state_dict["conv1.weight"].clone().repeat(1, in_channels, 1, 1, 1)
            )
            self.backbone.load_state_dict(state_dict, strict=False)

            # # MonAI resnet pretraining is buggy - does not let bias_downsample=False while pretrained=False
            # # and pretrained=True raises NotImplementedError
            # # this is a come-around

            # for name, param in self.backbone.named_parameters():
            #     if "downsample" in name and "bias" in name:
            #         param = zeros_like(param)
            #         param.requires_grad = False

        self.freeze_encoder = freeze_encoder_until is not None
        self.unfreeze_epoch = freeze_encoder_until

        if self.freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.use_tabular = num_tabular_features > 0
        self.fc2 = nn.Sequential(
            nn.Linear(out_size + num_tabular_features, (out_size // 4)),
            # nn.BatchNorm1d(32 * self._BUILDING_BLOCK.EXPANSION),
            nn.LeakyReLU(),
        )

        self.num_target_classes = num_target_classes
        if not isinstance(self.num_target_classes, list):
            self.num_target_classes = [self.num_target_classes]

        self.fc_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear((out_size // 4), num_class, bias=False),
                    nn.Flatten(start_dim=0) if num_class == 1 else nn.Identity(),
                )
                for num_class in self.num_target_classes
            ]
        )

    def forward(self, x: Tensor, tabular: Tensor = None) -> Tensor:
        if not len(x.shape) == 2:
            # otherwise, image embeddings precomputed
            x = self.backbone(x)

        if self.use_tabular:
            x = cat([x, tabular], dim=1)
        x = self.fc2(x)

        out = [head(x) for head in self.fc_heads]
        return out
    
    def get_embeddings(self,x: Tensor) -> Tensor:
        
        x = self.backbone(x)
        return x


class MonaiResNetMeta(type):
    def __call__(
        cls,
        architecture: str = "resnet50",
        pretrained_path: str = None,
        input_channels: int = 4,
        **kwargs
    ):
        if pretrained_path:
            model = getattr(nets, architecture)(spatial_dims=3, n_input_channels=1, **kwargs)
            print("Loading pretrained weights from: ", pretrained_path)
            pretrained_path = Path(pretrained_path).expanduser().resolve()
            state_dict = load(pretrained_path, map_location="cpu")["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            conv1_weight = model.conv1.weight.clone()
            model.conv1 = nn.Conv3d(
                input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False
            )
            with no_grad():
                model.conv1.weight = conv1_weight.repeat(1, input_channels, 1, 1, 1)
        else:
            model = getattr(nets, architecture)(
                spatial_dims=3, n_input_channels=input_channels, **kwargs
            )

        instance = deepcopy(model)

        return instance


class MonaiResnet(metaclass=MonaiResNetMeta):
    pass


## same thing in function form, if necessary
def monai_resnet(
    architecture: str = "resnet50", pretrained_path: str = None, input_channels: int = 4, **kwargs
):
    if pretrained_path:
        model = getattr(nets, architecture)(**kwargs)
        print("Loading pretrained weights from: ", pretrained_path)
        pretrained_path = Path(pretrained_path).expanduser().resolve()
        state_dict = load(pretrained_path, map_location="cpu")["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        conv1_weight = model.conv1.weight.clone()
        model.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        with no_grad():
            model.conv1.weight = conv1_weight.repeat(1, input_channels, 1, 1, 1)
    else:
        model = getattr(nets, architecture)(
            spatial_dims=3, n_input_channels=input_channels, **kwargs
        )

    return model
