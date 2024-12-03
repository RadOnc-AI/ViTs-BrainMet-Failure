import os
import json
from pathlib import Path
from typing import List
from numpy import append, nan_to_num, zeros, where, argmax, asarray, isnan, log, int64, float32
from math import ceil
from nibabel import load
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from monai.data import Dataset, PersistentDataset, CacheDataset, list_data_collate
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler, BatchSampler
from collections import Counter
from data.utils import StratifiedRandomSampler
import pandas as pd
import cc3d
import data.transforms as transforms




def get_datasets(cfg):
    """
    Read paths in config and read images, define transforms, split (k-fold CV or normal)
    """
    print("Creating datasets...")

    data_path = Path(cfg.datasets.parent_path).expanduser().resolve()
    train_path = data_path / "Training_Set"
    test_path = data_path / "Test_Set"

    # Read patient subdirectories

    train_patients = [patient for patient in sorted(train_path.glob("*/*")) if os.path.isdir(patient)]
    test_patients = [patient for patient in sorted(test_path.glob("*/*")) if os.path.isdir(patient)]

    # Read label csv
    train_test_csv = pd.read_csv(data_path / cfg.datasets.label_csv).set_index("ID")
    

    # Construct dataset as a list(dict) containing relevant MRI sequences
    train_data = []
    test_data = []
    events = []  # construct event occurence (0/1) to use in stratify


    for patient in train_patients:
        patient_id = f"{patient.parts[-1]}_{patient.parts[-2]}"
        try:
            labels = (
                train_test_csv.loc[patient_id, cfg.datasets.target_labels]
                .to_numpy()
                .astype(float32)
            )
        except KeyError:
            continue
            # new tables are mostly cleaned from patients with missing info - so just skip them
        if isnan(labels).any() or (labels < 0).any():
            # LF labels are always in 0,1
            # some LC are missing or negative -- remove them
            continue
        images = [
            str(list(patient.glob(f"*{seq_name}.nii.gz"))[0])
            for seq_name in cfg.datasets.sequences_to_use
            if seq_name != "seg"
        ]
        patient_dict = {"patient_id": patient_id, "labels": labels}
        patient_dict["images"] = images


        if "LF" in cfg.datasets.target_labels:
            idx = cfg.datasets.target_labels.index("LF")
            events.append(labels[idx])
        if "LRF" in cfg.datasets.target_labels:
            idx = cfg.datasets.target_labels.index("LRF")
            events.append(labels[idx])
        if "LeptoF" in cfg.datasets.target_labels:
            idx = cfg.datasets.target_labels.index("LeptoF")
            events.append(labels[idx])
        if "BF" in cfg.datasets.target_labels:
            idx = cfg.datasets.target_labels.index("BF")
            events.append(labels[idx])

        patient_dict["mask"] = str(list(patient.glob(f"*label.nii.gz"))[0])


        if cfg.datasets.use_tabular:
            tabular_data = train_test_csv.loc[patient_id].drop(labels=['LF','LC','LRF','LRC','LeptoF','LeptoC','BF','BC'])
            patient_dict["tabular"] = pd.to_numeric(tabular_data).to_numpy().astype(float32)
        else:
            patient_dict["tabular"] = asarray([0.0])  # placeholder

        train_data.append(patient_dict)

    for patient in test_patients:
        patient_id = f"{patient.parts[-1]}_{patient.parts[-2]}"
        try:
            labels = (
                train_test_csv.loc[patient_id, cfg.datasets.target_labels]
                .to_numpy()
                .astype(float32)
            )
        except KeyError:
            continue
        if isnan(labels).any() or (labels < 0).any():
            continue
        images = [
            str(list(patient.glob(f"*{seq_name}.nii.gz"))[0])
            for seq_name in cfg.datasets.sequences_to_use
            if seq_name != "seg"
        ]
        patient_dict = {"patient_id": patient_id, "labels": labels}
        patient_dict["images"] = images

        patient_dict["mask"] = str(list(patient.glob(f"*label.nii.gz"))[0])  # todo: filename


        if cfg.datasets.use_tabular:
            tabular_data = train_test_csv.loc[patient_id].drop(labels=['LF','LC','LRF','LRC','LeptoF','LeptoC','BF','BC'])
            patient_dict["tabular"] = pd.to_numeric(tabular_data).to_numpy().astype(float32)
        else:
            patient_dict["tabular"] = asarray([0.0])  # placeholder

        test_data.append(patient_dict)

    # Read transforms
    subtract_seq = transforms.Lambdad(["images"], lambda x: x)
    to_point_cloud = transforms.Lambdad(["images"], lambda x: x)

    train_transform_list = []
    for name, options in cfg.transforms.train.items():
        train_transform_list.append(getattr(transforms, name)(**options))

    test_transform_list = []
    for name, options in cfg.transforms.test.items():    
        test_transform_list.append(getattr(transforms, name)(**options))  # empty if no test transforms given

    

    image_keys = ["images"] #["t1c", "images"] if cfg.use_t1c_separate else ["images"]
    all_keys = [*image_keys, "mask"] if ("seg" in cfg.datasets.sequences_to_use and not cfg.training.joint_segmentation) else image_keys

    if (len(cfg.datasets.sequences_to_use) == 1) or (
        ("seg" in cfg.datasets.sequences_to_use) and (len(cfg.datasets.sequences_to_use) == 2)
    ):
        AddChannel = transforms.EnsureChannelFirstd(
            [*image_keys, "mask"], strict_check=False, channel_dim="no_channel"
        )
    else:
        AddChannel = transforms.EnsureChannelFirstd(
            "mask", strict_check=False, channel_dim="no_channel"
        )
    

    default_transforms = [
        transforms.LoadImageD(keys=[*image_keys, "mask"],image_only=False),
        transforms.Lambdad([*image_keys, "mask"], nan_to_num),
        # if using single MRI sequence expand to CHWD
        AddChannel,
        transforms.EnsureTyped(keys=[*image_keys, "mask"]),
        transforms.Compose(
            [
                transforms.ToTensord(keys=all_keys),
                transforms.ConcatItemsd(
                    keys=all_keys, name="images"
                ),  # does nothing if only images, elif seg is present concat
            ]
        ),
    ]
    if "Preprocessed" not in str(data_path):
        extra_transforms = [
            transforms.Orientationd(keys=["images", "mask"], axcodes="RAS"),
            transforms.ScaleIntensityRangePercentilesd(
                keys=image_keys,
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                relative=False,
                channel_wise=True,
            ),
        ]
        default_transforms = [*default_transforms[:-1], *extra_transforms, default_transforms[-1]]


    if not cfg.datasets.sequences_to_use == []:
        train_transforms = transforms.Compose(
            [*default_transforms[:-1], *train_transform_list, default_transforms[-1]]
        )
        test_transforms = transforms.Compose(
            [*default_transforms[:-1], *test_transform_list, default_transforms[-1]]
        )
    else:
        train_transforms = None
        test_transforms = None

    # Split
    if cfg.split.precomputed_splits:
        print("Using provided precomputed splits")
        splits_file = data_path / cfg.split.precomputed_splits
        with open(splits_file, "r") as f:
            precomputed_splits = json.load(f)
        train_val_splits = []
        for fold in precomputed_splits.values():
            train_val_splits.append(read_splits(train_data, fold))
    elif "overfit" not in cfg.name:
        if not cfg.split.cross_val:
            # nested in list for compatibility with kfolds
            train_split, val_split, train_events, val_events = train_test_split(
                train_data,
                events,
                train_size=cfg.split.train,
                random_state=cfg.general.seed,
                stratify=events,
            )
            train_val_splits = [((train_split, train_events), (val_split, val_events))]
        else:
            print("WARNING: Creating new splits for cross-validation.")
            kfold = StratifiedKFold(n_splits=cfg.split.cross_val, shuffle=False)
            train_val_splits = []
            for train_indices, val_indices in kfold.split(train_data, events):
                train_split = [train_data[i] for i in train_indices]
                train_events = [events[i] for i in train_indices]
                val_split = [train_data[i] for i in val_indices]
                val_events = [events[i] for i in val_indices]
                train_val_splits.append(((train_split, train_events), (val_split, val_events)))
    else:
        # test to overfit on same samples
        train_val_splits = [((train_data, events), (train_data, events))]

    return train_val_splits, test_data, train_transforms, test_transforms

def read_splits(train_data, fold):
    train_split = []
    train_events = []
    val_split = []
    for patient in train_data:
            if patient["patient_id"] in fold["train"]:
                train_split.append(patient)
                train_events.append(patient["labels"][0])
            elif patient["patient_id"] in fold["val"]:
                val_split.append(patient)
            else:
                continue 
    return ((train_split, train_events), (val_split, None)) # val events not used anyway


def get_dataloaders(cfg, train_val_data, test_data, train_transforms, test_transforms, fold=0):
    """
    Construct datasets based on the current split, return dataloaders
    """

    # extract data out of the splits
    train_data = train_val_data[fold][0][0]
    train_events = train_val_data[fold][0][1]
    val_data = train_val_data[fold][1][0]
    val_events = train_val_data[fold][1][1]

    # define datasets
    if cfg.datasets.cache_dir:
        cache = cfg.datasets.cache_dir.expanduser().resolve()
        cache = cache / cfg.general.run_id
        [
            dir.mkdir(parents=True, exist_ok=True)
            for dir in [cache, cache / "train", cache / "val", cache / "test"]
        ]

        print(f"Using persistent dataset cached in {cache}")
        train_dataset = PersistentDataset(
            data=train_data, transform=train_transforms, cache_dir=cache / "train"
        )
        val_dataset = PersistentDataset(
            data=val_data, transform=test_transforms, cache_dir=cache / "val"
        )
        test_dataset = PersistentDataset(
            data=test_data, transform=test_transforms, cache_dir=cache / "test"
        )
    elif cfg.datasets.cached_dataset:
        print(f"Using cached dataset")
        cache_train = float(cfg.general.train)
        train_dataset = CacheDataset(data=train_data, transform=train_transforms, cache_rate=cache_train, num_workers=cfg.loader.num_workers)
        val_dataset = CacheDataset(data=val_data, transform=test_transforms, cache_rate=cache_train, num_workers=cfg.loader.num_workers)
        # maybe no need to keep test in cache during training
        cache_test = float(not cfg.general.train)
        test_dataset = CacheDataset(data=test_data, transform=test_transforms, cache_rate=cache_test, num_workers=cfg.loader.num_workers)
    else:
        train_dataset = Dataset(data=train_data, transform=train_transforms)
        val_dataset = Dataset(data=val_data, transform=test_transforms)
        test_dataset = Dataset(data=test_data, transform=test_transforms)

    # define dataloaders
    train_dataloader_args = {
        "dataset": train_dataset,
        "num_workers": cfg.loader.num_workers,
        "collate_fn": list_data_collate,
    }
    if (not hasattr(cfg.loader, "stratified_batch")) and (not hasattr(cfg.loader, "oversampling_ones")):
        raise ValueError(
            "Wrong config. Please insert both 'stratified_batch' and 'oversampling_ones', but set False if not using"
        )
    
        
    if cfg.loader.stratified_batch:
        if cfg.loader.shuffle.train:            
            sampler = StratifiedRandomSampler(train_events, cfg.loader.batch_size.train)
            batch_sampler = BatchSampler(sampler, cfg.loader.batch_size.train, False)
        # else:
        #     sampler = SequentialSampler(train_dataset)
        
        train_dataloader_args["batch_sampler"] = batch_sampler
    elif cfg.loader.oversampling_ones:
        assert cfg.loader.shuffle.train, "Oversampling not supported without shuffling"
        
        weights = [(0.3, 0.7)[int(x)] for x in train_events]
        sampler = WeightedRandomSampler(weights, int(2 * len(weights)), replacement=True)
        batch_sampler = BatchSampler(sampler, cfg.loader.batch_size.train, False)
        train_dataloader_args["batch_sampler"] = batch_sampler
    else:
        
        train_dataloader_args["batch_size"] = (
            len(train_events) if cfg.loader.batch_size.train == "all" else cfg.loader.batch_size.train
        )
        train_dataloader_args["shuffle"] = cfg.loader.shuffle.train

    train_loader = DataLoader(**train_dataloader_args)

    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset) if cfg.loader.batch_size.test == "all" else cfg.loader.batch_size.test,
        num_workers=cfg.loader.num_workers,
        collate_fn=list_data_collate,
        shuffle=cfg.loader.shuffle.test,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset) if cfg.loader.batch_size.test == "all" else cfg.loader.batch_size.test,
        num_workers=cfg.loader.num_workers,
        collate_fn=list_data_collate,
        shuffle=cfg.loader.shuffle.test,
    )

    print("#" * 45 + " DataLoaders created " + "#" * 45)
    print(
        f"Training images: {len(train_dataset)}",
        f"Validation images: {len(val_dataset)}",
        f"Test images: {len(test_dataset)}",
        sep="\n",
    )

    return train_loader, val_loader, test_loader


def find_nearest_dividend(number, divisor):
    while number % divisor != 0:
        number = ceil(number) + 1
    else:
        return number, number / divisor

