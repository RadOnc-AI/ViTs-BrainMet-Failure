from torch import nn, sqrt
from torch.nn.functional import mse_loss

class RMSELoss(nn.Module):
    """
    Add root-MSE Functionality
    """

    def __init__(
        self,
        reduction: str = 'mean',
        eps: float = 1e-6,
    ):

        super().__init__()
        # self.mse = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
        self.eps = eps

    def forward(self, yhat, y):
        loss = sqrt(mse_loss(yhat,y,reduction=self.reduction) + self.eps)
        return loss
