import numpy as np
from torch import Tensor, float32, functional, mean, nn, tensor


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Binary Cross Entropy Loss.

    Creates a criterion that measures the Binary Cross Entropy
    between the target and the output. This class is an override of
    torch.nn.BCEWithLogitsLoss, which flattens the input before calculating the loss.
    """

    def __init__(
        self,
        class_weight: tuple[float, float] = None,
        weight: Tensor = None,
        reduction: str = "mean",
    ):
        """Create a new BCELoss instance.

        Args:
            class_weight: Weights for each class.
            weight: a manual rescaling weight given to the loss of each batch
                element. If given, has to be a Tensor of size `nbatch`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no
                reduction will be applied, ``'mean'``: the sum of the output
                will be divided by the number of elements in the output,
                ``'sum'``: the output will be summed.
        """
        if class_weight is not None and weight is not None:
            raise ValueError(
                "Specifying class weights and weight not allowed."
            )
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = np.array(class_weight) / sum(class_weight)
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            input: Input tensor.
            target: Target tensor.

        Returns:
            loss: Loss
        """
        if self.class_weight is not None:
            weight = tensor(
                [self.class_weight[int(t == 1)] for t in target],
                dtype=float32,
            ).to(input.device)
            loss = functional.F.binary_cross_entropy_with_logits(
                input,
                target.type(float32),
                reduction="none",
            )
            return mean(weight * loss)
        weight = self.weight
        return functional.F.binary_cross_entropy_with_logits(
            input,
            target.type(float32),
            reduction=self.reduction,
        )
