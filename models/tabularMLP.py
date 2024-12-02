from typing import Type, Union, List

import torch.nn.functional as F
from torch import Tensor, cat, flatten, nn, zeros, exp, log, load
from torch.autograd import Variable

class TabularMLP(nn.Module):
    
    def __init__(
            self,
            dropout: float = 0.5,
            num_tabular_features: int = 19,
            hidden_sizes: List[int] = [30],
            num_target_classes: List[int] = [1],
            **kwargs
            ) -> None:
        super().__init__()

        self.dropout = dropout
        self.freeze_encoder = False
        self.num_target_classes = num_target_classes

        
        self.first = nn.Sequential(
            nn.Linear(num_tabular_features,hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Mish()
        )

        self.hiddens = nn.Sequential(
            *[nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(hidden_sizes[i-1],hidden_sizes[i]),
                nn.BatchNorm1d(hidden_sizes[i]),
                nn.Mish()
            ) for i in range(1,len(hidden_sizes))]
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(hidden_sizes[-1],num_class),
                nn.Flatten(start_dim=0) if num_class==1 else nn.Identity()
            )
            for num_class in self.num_target_classes
        ])

    def forward(self,img,x):
        """
        input-output compatible with image models in CoxTrainingBase
        """
        x = self.first(x)
        x = self.hiddens(x)
        out = [head(x) for head in self.heads]

        return out
    
