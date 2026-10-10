from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .utils import seed_worker

REQUIRED_COLUMNS = {"path", "identity", "stage", "split", "group_id"}
SPLIT_ALIASES = {"val": "validation", "valid": "validation"}


@dataclass(frozen=True)
class DatasetSpecification:
    root: Path
    stages: Sequence[str]
    num_classes: int
    expected_per_identity: Mapping[str, int]
    strict_counts: bool


def read_manifest(path: Union[str, Path]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError("Missing manifest columns: {}".format(sorted(missing)))
    frame = frame.copy()
    frame["identity"] = frame["identity"].astype(str)
    frame["stage"] = frame["stage"].astype(str)
    frame["split"] = frame["split"].astype(str).str.lower().replace(SPLIT_ALIASES)
    frame["group_id"] = frame["group_id"].astype(str)
    if "acquisition_type" not in frame:
        frame["acquisition_type"] = "unspecified"
    frame["acquisition_type"] = frame["acquisition_type"].astype(str)
    return frame


def identity_mapping(frame: pd.DataFrame) -> Dict[str, int]:
    identities = sorted(frame["identity"].unique().tolist())
    return {identity: index for index, identity in enumerate(identities)}


def validate_manifest(frame: pd.DataFrame, specification: DatasetSpecification) -> Dict:
    allowed_splits = {"train", "validation", "test"}
    unknown_splits = set(frame["split"]) - allowed_splits
    if unknown_splits:
        raise ValueError("Unknown split labels: {}".format(sorted(unknown_splits)))
    missing_stages = set(specification.stages) - set(frame["stage"])
    if missing_stages:
        raise ValueError("Missing stages: {}".format(sorted(missing_stages)))

    identities_by_stage = {
        stage: set(frame.loc[frame["stage"] == stage, "identity"])
        for stage in specification.stages
    }
    reference = identities_by_stage[specification.stages[0]]
    if len(reference) != specification.num_classes:
        raise ValueError(
            "Expected {} identities but found {}".format(
                specification.num_classes, len(reference)
            )
        )
    for stage, identities in identities_by_stage.items():
        if identities != reference:
            raise ValueError("The identity set at {} differs from T1".format(stage))

    leakage = []
    group_columns = ["stage", "acquisition_type", "group_id"]
    for key, group in frame.groupby(group_columns, sort=False):
        splits = sorted(set(group["split"]))
        if len(splits) > 1:
            leakage.append({"group": key, "splits": splits})
    if leakage:
        raise ValueError(
            "Acquisition groups occur in multiple subsets: {}".format(leakage[:5])
        )

    if specification.strict_counts:
        counts = frame.groupby(["stage", "split", "identity"]).size()
        errors: List[str] = []
        for stage in specification.stages:
            for split, expected in specification.expected_per_identity.items():
                try:
                    observed = counts.loc[(stage, split)]
                except KeyError:
                    errors.append("{}/{}: subset is missing".format(stage, split))
                    continue
                if len(observed) != specification.num_classes:
                    errors.append(
                        "{}/{}: expected {} identities but found {}".format(
                            stage, split, specification.num_classes, len(observed)
                        )
                    )
                invalid = observed[observed != int(expected)]
                if len(invalid):
                    errors.append(
                        "{}/{}: {} identities do not have {} images".format(
                            stage, split, len(invalid), expected
                        )
                    )
        if errors:
            raise ValueError("; ".join(errors))

    missing_paths = []
    for relative_path in frame["path"]:
        path = Path(str(relative_path))
        path = path if path.is_absolute() else specification.root / path
        if not path.is_file():
            missing_paths.append(str(path))
            if len(missing_paths) == 5:
                break
    if missing_paths:
        raise FileNotFoundError("Image files are missing: {}".format(missing_paths))

    return {
        "images": int(len(frame)),
        "identities": int(len(reference)),
        "stages": list(specification.stages),
        "cross_subset_group_overlap": 0,
    }


def build_transforms(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    augmentation: Mapping[str, float],
):
    training = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(
                p=float(augmentation["horizontal_flip_probability"])
            ),
            transforms.RandomRotation(
                float(augmentation["rotation_degrees"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=float(augmentation["brightness"]),
                contrast=float(augmentation["contrast"]),
                saturation=float(augmentation["saturation"]),
                hue=float(augmentation["hue"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    replay = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(
                p=float(augmentation["horizontal_flip_probability"])
            ),
            transforms.RandomRotation(
                float(augmentation["rotation_degrees"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=float(augmentation["brightness"]),
                contrast=float(augmentation["contrast"]),
                saturation=float(augmentation["saturation"]),
                hue=float(augmentation["hue"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return training, evaluation, replay


class SheepFaceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        root: Union[str, Path],
        label_map: Mapping[str, int],
        transform,
        image_size: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.root = Path(root)
        self.label_map = dict(label_map)
        self.transform = transform
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def image_path(self, index: int) -> Path:
        path = Path(str(self.frame.iloc[index]["path"]))
        return path if path.is_absolute() else self.root / path

    def raw_resized_uint8(self, index: int) -> np.ndarray:
        with Image.open(self.image_path(index)) as handle:
            image = handle.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        return np.asarray(image, dtype=np.uint8).copy()

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.image_path(index)) as handle:
            image = self.transform(handle.convert("RGB"))
        return {
            "image": image,
            "label": torch.tensor(
                self.label_map[str(row["identity"])], dtype=torch.long
            ),
            "identity": str(row["identity"]),
            "stage": str(row["stage"]),
            "path": str(self.image_path(index)),
        }


def select_rows(frame: pd.DataFrame, stage: Optional[str], split: str) -> pd.DataFrame:
    mask = frame["split"].eq(split)
    if stage is not None:
        mask &= frame["stage"].eq(stage)
    return frame.loc[mask].copy()


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=workers > 0,
    )
