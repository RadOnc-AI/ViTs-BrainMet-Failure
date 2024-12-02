from torch import nn, Tensor, zeros_like, arange, ones, logical_not, sum, mean, clamp, cumprod, cat
from monai.data.meta_tensor import MetaTensor
import torch.nn.functional as F


class CoxPHLoss(nn.Module):
    """
    Re-implementation of cox_ph_loss from https://github.com/havakv/pycox/blob/69940e0b28c8851cb6a2ca66083f857aee902022/pycox/models/loss.py#L425
    Basic Cox proportional hazards model loss based on DeepSurv
    https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but computationally simpler and faster.
    """

    def __init__(
            self,
            eps: float = 1e-6,
            num_intervals: int = 20, 
            interval_duration: float = 6.0,
            ticks: list = None, ):
        super().__init__()
        self.eps = eps
        # for combined usage with DiscreteSurvLoss
        if ticks:
            self.intervals = Tensor(ticks[:-1])
            self.num_intervals = len(self.intervals)
        else:
            self.num_intervals = num_intervals
            self.intervals = arange(self.num_intervals) * interval_duration

    def forward(self, preds: Tensor, events: Tensor, durations: Tensor):
        """
        preds: hazard predictions of model (h)
        events: LF
        durations: LC
        everything should be flattened
        """

        if isinstance(preds, MetaTensor):
            preds = preds.as_tensor()
        # preds = preds.sigmoid()
        
        if not len(preds.shape) == 1:
            # discrete model outputs, coxPH as auxiliary loss
            assert preds.shape[1] == self.num_intervals
            preds = interval_preds_to_hazard(preds, self.intervals)

        # sort into shortest time to longest
        sort_idx = durations.sort(descending=True)[1]
        events = events[sort_idx].float().flatten()  # 0.0 or 1.0
        preds = preds[sort_idx].flatten()

        gamma = preds.max()
        log_cumsum_preds = preds.sub(gamma).exp().cumsum(dim=0).add(self.eps).log().add(gamma)
        return -preds.sub(log_cumsum_preds).mul(events).sum().div(events.sum() + self.eps)


class DiscreteSurvLoss(nn.Module):
    """
    You can provide ticks for custom intervals, or construct intervals of equal lengths automatically 
    by only providing num_intervals
    """
    def __init__(
            self, 
            eps: float = 1e-6,
            masked: bool = True, 
            num_intervals: int = 20, 
            interval_duration: float = 6.0,
            ticks: list = None, 
            reduction: str = "mean",
            event_weight: float = None):
        super().__init__()
        self.eps = eps
        self.masked = masked
        reduction = reduction.lower()
        assert reduction in ["mean", "sum"]
        if reduction == 'mean':
            self.reduction = lambda x: mean(x, dim=0)
        else:
            self.reduction = lambda x: sum(x, dim=0)
        
        if event_weight:
            self.event_weights = Tensor([event_weight])
        else:
            self.event_weights = Tensor([1.])
        if ticks:
            self.intervals = Tensor(ticks[:-1])
            self.num_intervals = len(self.intervals)
        else:
            self.num_intervals = num_intervals
            self.intervals = arange(self.num_intervals) * interval_duration


    def forward(self, preds: Tensor, events: Tensor, durations: Tensor):
        """
        N: number of discrete intervals

        preds: B x N
        events: B x 1 binary coded event
        survivals: B x 1 time to event (or censoring)

        events and survivals will be encoded to time interval
        """
        assert preds.shape[1] == self.num_intervals
        if isinstance(preds, MetaTensor):
            preds = preds.as_tensor()

        mask = self.intervals.to(preds.device).expand(durations.shape[0], -1) <= durations.view(-1, 1)
        mask = mask.float()
        durations = mask.argmin(dim=1).sub(1).long() # the interval of event/censoring - basically the last index of 111 masking
        durations[durations == -1] = len(durations) # trick to change to last index, -1 doesn't work with one_hot
        # durations = ones(mask.size(0),).long().to(mask.device) 
        # indices = mask.nonzero()
        # durations[indices[:, 0]] = indices[:, 1]
        survivals = F.one_hot(durations, num_classes=self.num_intervals).to(preds.device) * events.view(-1, 1)
        
        #preds = preds
        # if self.masked:
        #     mask = survivals.bool().logical_not().float() + events
        # else:
        # mask = ones_like(events)

        losses = F.binary_cross_entropy_with_logits(
            preds, 
            survivals.float(),
            mask, 
            reduction="none", 
            pos_weight=self.event_weights.expand(preds.shape[1]).to(preds.device))
        # sum or mean over patients, then sum over all time intervals
        
        return self.reduction(losses).sum()
        # if self.reduction == "mean":
        #     return losses.mean(dim=0).sum()
        # elif self.reduction == "sum":
        #     return losses.sum(dim=0).sum()


class GensheimerLoss(nn.Module):
    def __init__(
            self, 
            eps: float = 1e-6,
            num_intervals: int = 20, 
            interval_duration: float = 6.0,
            ticks: list = None, 
            reduction: str = "mean",
            event_weight: float = None):
        super().__init__()
        self.eps = eps
        reduction = reduction.lower()
        assert reduction in ["mean", "sum"]
        if reduction == 'mean':
            self.reduction = mean
        else:
            self.reduction = sum

        if event_weight:
            self.event_weights = event_weight
        else:
            self.event_weights = 1.0
        if ticks:
            self.intervals = Tensor(ticks[:-1])
            self.num_intervals = len(self.intervals)
        else:
            self.num_intervals = num_intervals
            self.intervals = arange(self.num_intervals) * interval_duration

    def forward(self, preds: Tensor, events: Tensor, durations: Tensor):
        """

        """

        assert preds.shape[1] == self.num_intervals
        if isinstance(preds, MetaTensor):
            preds = preds.as_tensor()

        mask = self.intervals.to(preds.device).expand(durations.shape[0], -1) <= durations.view(-1, 1)
        mask = mask.float()
        durations = mask.argmin(dim=1).sub(1).long() # the interval of event/censoring - basically the last index of 111 masking
        durations[durations == -1] = self.num_intervals - 1 # trick to change to last index, -1 doesn't work with one_hot
        # durations = ones(mask.size(0),).long().to(mask.device) 
        # indices = mask.nonzero()
        # durations[indices[:, 0]] = indices[:, 1]
        survivals = F.one_hot(durations, num_classes=self.num_intervals).to(preds.device) * events.view(-1, 1)
        
        mask = logical_not(mask).float()
        preds = preds.sigmoid()
        weights = survivals * self.event_weights
        
        log1 = clamp(1 + mask * (preds - 1),self.eps,None).log()
        log2 = clamp(1 - survivals * preds,self.eps,None).log() * weights #penalizing
        summed = (log1 + log2).sum(dim=1)

        return -1 * self.reduction(summed)
        # if self.reduction == "mean":
        #     return -1 * summed.mean()
        # elif self.reduction == "sum":
        #     return -1 * summed.sum()

def interval_preds_to_hazard(preds: Tensor, breaks: Tensor):
    breaks = breaks.to(preds.device)
    prev = cumprod((1.0 - preds[:, :-1]), 1)  # prod_l (1 - h_l)
    prev = cat([ones((preds.shape[0], 1), device=preds.device), prev], 1)
    surv_times = sum(breaks * preds * prev, 1)
    surv_times = (surv_times - 0) / (breaks.max() - 0)
    pred_hazards = 1 - surv_times

    return pred_hazards



if __name__ == "__main__":
    lss = DiscreteSurvLoss()
