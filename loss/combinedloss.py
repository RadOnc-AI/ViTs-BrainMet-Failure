from typing import List, Optional, Dict
from torch import nn, tensor, float32

import loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        cfg,
        loss_instances: Dict[str, dict],
        loss_weights: Optional[List[float]] = None,
        repeat_inputs: bool = False,
    ):
        super().__init__()
        if loss_weights:
            assert len(loss_instances) == len(loss_weights)
            self.loss_weights = loss_weights.copy()
        else:
            self.loss_weights = [1.0 for _ in range(len(loss_instances))]

        self.loss_instances = nn.ModuleList([getattr(loss, name)(**opt) for name, opt in loss_instances.items()])
        self.loss_names = [name for name in loss_instances]

        self.repeat_inputs = repeat_inputs
        if len(cfg.metrics) != len(self.loss_instances):
            assert repeat_inputs, f"If not using multiple losses for same target, number of metrics ({len(cfg.metrics)}) must match number of losses ({len(self.loss_instances)})"

    def forward(self, inputs):
        # inputs = list(inputs)
        loss_vals = []
        total_loss = tensor(0, dtype=float32) #.to(inputs[0][0].device)
        for i, loss_fn in enumerate(self.loss_instances):
            loss_input = inputs[0] if self.repeat_inputs else inputs[i]
            loss_val = loss_fn(*loss_input)
            total_loss = total_loss + (self.loss_weights[i] * loss_val)
            loss_vals.append(loss_val.detach())

        return total_loss, loss_vals

    def __len__(self):
        return len(self.loss_instances)
