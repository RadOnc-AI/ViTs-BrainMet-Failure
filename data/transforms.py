"""
Package that extends MonAI transforms with custom ones
"""
from monai.config import KeysCollection
import nibabel as nib
import numpy as np
from monai.transforms import *
from typing import List, Union
from torch import (
    Tensor,
    absolute,
    cat,
    from_numpy,
    logical_and,
    logical_or,
    stack,
    subtract,
    where,
)
import torch.nn.functional as F
from itertools import chain
import cc3d


class CropToMask(MapTransform):
    """
    Crop the images to a region-of-interest bounding box defined by the segmentation masks.

        keys: datadict key for images that the transform should be applied to
        mask_key: datadict key for the segmentation masks to be used as a look-up
        use_class: which class in the mask to be cropped to
        margin: optional margin to extend the bounding box in each direction
    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str,
        use_class: Union[int, List[int]] = 1,
        margin: int = 0,
        min_dim: tuple[int, int, int] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        assert mask_key is not None
        self.mask_key = mask_key
        self.margin = margin
        self.min_dim = min_dim

        if isinstance(use_class, int):
            self.use_class = [use_class for _ in range(len(self.keys))]
        elif isinstance(use_class, list):
            assert len(use_class) == len(
                self.keys
            ), "If provided separately for each key, use_class list needs to have same length as keys."
            self.use_class = use_class
        assert {1, 2}.intersection(
            set(self.use_class)
        ), "We assume we have 2 classes, i.e. edema, gross tumor volume"

    def __call__(self, data):
        # only load the mask image for the first time, then store bbox info in datadict for next epochs
        if "crop_bounding_box:1" not in data:
            mask = (
                nib.load(data[self.mask_key]).get_fdata()
                if isinstance(data[self.mask_key], str)
                else data[self.mask_key].squeeze(0).numpy()
            )  # mask is being used and already loaded by LoadImaged
            data["crop_bounding_box:1"], data["pad_bounding_box:1"] = CropToMask.get_bbox(
                mask, self.margin, self.min_dim, 1
            )
        if "crop_bounding_box:2" not in data:
            mask = (
                nib.load(data[self.mask_key]).get_fdata()
                if isinstance(data[self.mask_key], str)
                else data[self.mask_key].squeeze(0).numpy()
            )  # mask is being used and already loaded by LoadImaged
            data["crop_bounding_box:2"], data["pad_bounding_box:2"] = CropToMask.get_bbox(
                mask, self.margin, self.min_dim, 2
            )

        for key, use_class in zip(self.keys, self.use_class):
            bbox = data[f"crop_bounding_box:{use_class}"]
            padding = [ i for tuples in data[f"pad_bounding_box:{use_class}"]
                for i in list(reversed(tuples))
            ]
            if key in data:
                data[key] = data[key][
                    :,
                    int(bbox[0][0]) : int(bbox[0][1]) + 1,
                    int(bbox[1][0]) : int(bbox[1][1]) + 1,
                    int(bbox[2][0]) : int(bbox[2][1]) + 1,
                ]
                # padding =
                padding.reverse()
                data[key] = F.pad(data[key], tuple(padding), "constant", 0)
            elif not self.allow_missing_keys:
                raise KeyError(f"Specified key '{key}' not found in data.")

        return data

    @staticmethod
    def get_bbox(
        mask: np.array, margin: int = 0, min_dim: tuple[int, int, int] = None, use_class: int = 1
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        img = mask.copy()
        if use_class == 1:
            img[np.where(img != 0)] = 1  # merge GTV into edema
        else:
            img[np.where(img != 2)] = 0  # turn edema into background

        xmin, xmax = np.where(np.any(img, axis=(1, 2)))[0][[0, -1]]
        ymin, ymax = np.where(np.any(img, axis=(0, 2)))[0][[0, -1]]
        zmin, zmax = np.where(np.any(img, axis=(0, 1)))[0][[0, -1]]

        bbox = ((xmin, xmax), (ymin, ymax), (zmin, zmax))

        new_bbox = []
        to_pad = []
        for i in range(3):
            minv, maxv = (
                bbox[i][0],
                bbox[i][1],
            )
            current_size = maxv - minv + 1
            # center = int(np.rint(np.mean([minv, maxv])))
            if not min_dim:
                cmargin = 0
            else:
                cmargin = (min_dim[i] - current_size) / 2  # int(np.floor(min_dim[i] / 2))
            minv = int(np.floor(minv - cmargin - margin))
            pad_min = abs(min(minv, 0))
            minv = max(minv, 0)
            maxv = int(np.floor(maxv + cmargin + margin))
            pad_max = max(maxv - img.shape[i] + 1, 0)
            maxv = min(maxv, img.shape[i])
            new_bbox.append((minv, maxv))
            mean = np.mean([pad_min, pad_max])
            pad_min, pad_max = int(np.floor(mean)), int(np.ceil(mean))  # pad in both dimensions
            to_pad.append((pad_min, pad_max))
        bbox = tuple(new_bbox)  # type: ignore

        # bbox = tuple(tuple(np.clip((minv - margin, maxv + margin), 0, img.shape[i])) for i, (minv, maxv) in enumerate(bbox))

        return bbox, to_pad

class MultiChannelBasedOnBratsClassesToLabels(Transform):
    """
    Convert multi-channel predictions to single channel labels
    """

    def __call__(self, outputs: Tensor):
        # ET is label 3
        single_channel = where(outputs[2] == 1, 3, 0)
        # TC is label 3 and 1 merged -> exclude ET from TC to find 1
        single_channel[logical_and(~(outputs[2] == 1), outputs[0] == 1)] = 1
        # WT is all labels combined -> extract TC from WT
        single_channel[logical_and(~(outputs[0] == 1), outputs[1] == 1)] = 2

        return single_channel


class FindBlobs(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        new_keys: KeysCollection,
        allow_missing_keys: bool = True,
        channel_wise: bool = True,
    ):
        super().__init__(keys, allow_missing_keys)
        if new_keys:
            new_keys = [new_keys] if not isinstance(new_keys, list) else new_keys
            assert len(self.keys) == len(new_keys)
        else:
            new_keys = self.keys
        self.key_mapper = dict(zip(self.keys, new_keys))
        self.channel_wise = channel_wise

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                if self.channel_wise:
                    results = []
                    for channel in d[key]:
                        channel = np.where(channel.numpy().copy() == 2, 1, 0)
                        components = cc3d.connected_components(channel.numpy()).astype("float16")
                        results.append(from_numpy(components))
                    results = stack(results, axis=0).float()
                    d[self.key_mapper[key]] = results
                else:
                    raise NotImplementedError
            elif self.allow_missing_keys:
                pass
            else:
                raise KeyError
        return d    

class SubtractionSequence(MapTransform):
    """
    Subtract given channel indices (ORDER VARIANT)
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        channel_indices: List[int] = [],
        operation: str = "squared",
    ):
        super().__init__(keys, allow_missing_keys)
        self.channel_indices = channel_indices
        self.operation = {
            "abs": absolute,
            "squared": lambda x: x**2,
            "none": lambda x: x,
        }[operation]

    def _compute_difference(self, sequences: list):
        return self.operation(subtract(*sequences))

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                assert (
                    len(data[key].shape) == 4
                ), "the image should be multi-channel with stacked sequences"
                sequences = [data[key][i, ...] for i in self.channel_indices]
                diff = self._compute_difference(sequences)
                data[key] = cat([data[key], diff.unsqueeze(0)])
            else:
                raise KeyError

        return data

class ExtractLargestMetastasis(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str,
        use_class: Union[int, List[int]] = 1,
        margin: int = 0,
        min_dim: tuple[int, int, int] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        assert mask_key is not None
        self.mask_key = mask_key

    def __call__(self, data):
        mask = data[self.mask_key].squeeze(0).numpy()
        edema = np.where(mask.copy() > 0, 1, 0)  # merge tumor and edema
        metastases = np.where(mask.copy() == 2, 1, 0)  # take metastases only

        # perform CCA to separate distinct metastases, take largest one and then find the edema for it
        # largest edema does not always (in 6 patients) contain largest metastasis
        largest_metastasis = cc3d.largest_k(metastases, k=1)
        edemas = cc3d.connected_components(edema)

        edema_index = find_largest_intersection(largest_metastasis, edemas)
        mask = np.where(
            edemas.copy() == edema_index, mask, 0
        )  # select edema and metastasis that belongs to the largest metastasis
        data[self.mask_key] = from_numpy(mask).unsqueeze(0)

        return data


def find_largest_intersection(largest_tumor, edemas):
    volumes = [0]
    for _, image in cc3d.each(edemas, binary=True, in_place=True):
        volumes.append((largest_tumor * image).sum())
    return np.asarray(volumes).argmax()


class ToPointCloud(MapTransform):
    """
    Args:
        as_list: if True, return the batch as a list of tensors, otherwise as a single tensor
                 needs to be True if images do not have the standard shape
    """
    def __init__(
            self, 
            keys: KeysCollection, 
            allow_missing_keys: bool = False,
            as_list: bool = False, 

        ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.as_list = as_list

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            if key in d:

                coordinates = (d[key][0] + 1e-6).nonzero() 
                values = d[key].permute(1,2,3,0).view(-1,3)
                
                d[key] = cat((coordinates, values), dim=1)
                if self.as_list:
                    d[key] = [d[key]]
            elif self.allow_missing_keys:
                pass
            else:
                raise KeyError
        return d