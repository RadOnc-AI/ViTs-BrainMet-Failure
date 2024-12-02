from typing import Iterator, Sized
from torch.utils.data import Sampler, BatchSampler
from sklearn.model_selection import StratifiedShuffleSplit
from numpy import hstack, array, split, arange
from math import ceil

import torch

    
class StratifiedRandomSampler(Sampler):

    def __init__(self, class_labels: list, batch_size: int ) -> None:
        super().__init__(data_source=None)

        self.class_labels = [int(x) for x in class_labels]
        self.n_splits = int(len(class_labels) / batch_size)

    def __len__(self):
        return len(self.class_labels)

    def _gen_sample_array(self):

        s  = StratifiedShuffleSplit(n_splits=self.n_splits,test_size=0.5)
        X = list(range(len(self.class_labels))) # placeholder, just indices
        s.get_n_splits(X,self.class_labels)

        train_index, test_index = next(s.split(X,self.class_labels))
    
        return hstack([train_index,test_index])

    def __iter__(self):
        return iter(self._gen_sample_array())


def calc_center_of_mass(x3d: torch.Tensor,
                        filter_nans: bool = True) -> torch.Tensor:
    """
    Returns centre of mass for each 3d Volume in batch.
    In: B x (C x) D x H x W
    Out: B x 3
    """
    assert x3d.dim() == 4, 'MRI must be 4D: BxDxHxW'
    x3d = x3d.float()
    n_x, n_y, n_z = x3d.shape[-3:]
    ii, jj, kk = torch.meshgrid(
        torch.arange(n_x),
        torch.arange(n_y),
        torch.arange(n_z),
        indexing='ij',
    )
    coords = torch.stack(
        [ii.flatten(), jj.flatten(), kk.flatten()],
        dim=-1).float().to(x3d.device)
    vmin = torch.min(x3d)
    vmax = torch.max(x3d)
    if vmax.allclose(vmin) and not vmax.allclose(torch.zeros(1)):
        # everything is tumor
        x3d_norm = torch.ones_like(x3d)
    else:
        x3d_norm = (x3d - vmin) / (vmax - vmin)
    if x3d_norm.dim() == 5:
        x3d_norm = (x3d_norm > 0).any(dim=1).float()
    x3d_list = torch.flatten(x3d_norm, start_dim=-3).unsqueeze(-1)
    brainmask_approx = (x3d_list > 0.).all(dim=0).squeeze()
    coords = coords[brainmask_approx]
    x3d_list = x3d_list[:, brainmask_approx]
    total_mass = torch.sum(x3d_list, dim=1)
    centre_of_mass = torch.sum(x3d_list * coords, dim=1) / total_mass
    if torch.any(torch.isnan(centre_of_mass)) and filter_nans:
        # backup method
        print(
            'Centre of mass contains NaN. Using backup method. every entry is zero, this should not happen.'
        )
        isna_mask = torch.isnan(centre_of_mass).any(dim=1)
        n_isna = isna_mask.sum()
        mean_coord = torch.tensor([n_x / 2, n_y / 2, n_z / 2],
                                  device=x3d.device,
                                  dtype=x3d.dtype)
        centre_of_mass[isna_mask] = mean_coord.unsqueeze(0).repeat(n_isna, 1)
    return centre_of_mass