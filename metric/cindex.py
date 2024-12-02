from torch import Tensor, arange, cumprod, sum, cat, ones
from torchmetrics import Metric
from sksurv.metrics import concordance_index_censored
from sksurv.exceptions import NoComparablePairException
from monai.data.meta_tensor import MetaTensor


class CIndex(Metric):
    def __init__(
            self,
            interval_duration: float = None, 
            ticks: list = None,
            loss_fn: str = "DiscreteSurvLoss",
            **kwargs):
        super().__init__(**kwargs)

        self.interval_duration = interval_duration
        if ticks:
            self.breaks = Tensor(ticks[1:]).view(1, -1)
        self.add_state("pred_hazards", default=Tensor(0))
        self.add_state("durations", default=Tensor(0))
        self.add_state("events", default=Tensor(0))

        # depending on which discrete loss, either switch survival
        # preds to hazard or do nothing
        if loss_fn == "GensheimerLoss":
            self.inversion = lambda x: 1 - x
        else:
            self.inversion = lambda x: x 

    def update(
        self,
        preds: Tensor,
        events: Tensor,
        durations: Tensor,
    ):
        if isinstance(preds, MetaTensor):
            preds = preds.as_tensor()

        if len(preds.shape) == 1:
            self.pred_hazards = preds.flatten()  # already flattened
        else:
            preds = preds.sigmoid()  # into hazard prob
            preds = self.inversion(preds)
            if not hasattr(self, "breaks"):
                self.breaks = (
                    arange(1, preds.shape[1] + 1, device=preds.device) * self.interval_duration
                )  # interval end points
            else:
                self.breaks = self.breaks.to(preds.device)
            prev = cumprod((1.0 - preds[:, :-1]), 1)  # prod_l (1 - h_l)
            prev = cat([ones((preds.shape[0], 1), device=preds.device), prev], 1)
            surv_times = sum(self.breaks * preds * prev, 1)
            surv_times = (surv_times - 0) / (self.breaks.max() - 0)
            self.pred_hazards = 1 - surv_times


        self.events = events.bool().flatten()
        self.durations = durations.flatten()

        # sanity check
        # from lifelines.utils import concordance_index
        # cindex = concordance_index(self.durations.cpu().numpy(),surv_times.cpu().numpy(),self.events.cpu().numpy())

    def compute(self):
        try:
            return concordance_index_censored(
                self.events.cpu().numpy(),
                self.durations.cpu().numpy(),
                self.pred_hazards.cpu().numpy(),
            )[0]
        except NoComparablePairException:
            return 0.5
        