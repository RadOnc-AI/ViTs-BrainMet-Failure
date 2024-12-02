import os
import torch
import numpy as np

from sklearn.metrics import matthews_corrcoef
from torchmetrics import Metric


class MCC(Metric):
    
    def __init__(self, threshold:float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.threshold = threshold

        self.add_state("preds", default=torch.Tensor(0))
        self.add_state("targets", default=torch.Tensor(0))

    def update(self,pred: torch.Tensor, target: torch.Tensor):
        
        # official Tochmetrics logits to prob conversion
        if pred.is_floating_point():
            if not ((0 <= pred) * (pred <= 1)).all():
                # preds is logits, convert with sigmoid
                pred = pred.sigmoid()
            #preds = preds > threshold

        self.preds = torch.where((pred >= self.threshold),1,0) # works for both binarized or probability outputs
        self.targets = target

    def compute(self):
        return matthews_corrcoef(self.preds.cpu().numpy(), self.targets.cpu().numpy())