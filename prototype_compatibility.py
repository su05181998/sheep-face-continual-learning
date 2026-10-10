import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pgrcl.config import load_config
from pgrcl.data import (
    SheepFaceDataset,
    build_transforms,
    identity_mapping,
    read_manifest,
    select_rows,
)
from pgrcl.model import build_cgsfnet
from pgrcl.utils import load_torch_file, select_device


@torch.no_grad()
def prototypes(model, dataset, label_map, device, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dimension = model.embedding_dim
    sums = torch.zeros(len(label_map), dimension, device=device)
    counts = torch.zeros(len(label_map), device=device)
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        features = model.forward_features(images, normalize=True)
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    return F.normalize(sums / counts.unsqueeze(1), p=2, dim=1)


def load_model(config, checkpoint, device):
    model = build_cgsfnet(config["model"]).to(device)
    payload = load_torch_file(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-version prototype compatibility"
    )
    parser.add_argument("--config", default="configs/manuscript.yaml")
    parser.add_argument("--earlier-checkpoint", required=True)
    parser.add_argument("--later-checkpoint", required=True)
    parser.add_argument("--stage", choices=["T1", "T2", "T3"], required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    device = select_device(arguments.device)
    frame = read_manifest(Path(config["_project_root"]) / config["data"]["manifest"])
    label_map = identity_mapping(frame)
    _, evaluation, _ = build_transforms(
        config["data"]["image_size"],
        config["data"]["mean"],
        config["data"]["std"],
        config["augmentation"],
    )
    dataset = SheepFaceDataset(
        select_rows(frame, arguments.stage, "test"),
        Path(config["_project_root"]) / config["data"]["root"],
        label_map,
        evaluation,
        config["data"]["image_size"],
    )
    first = prototypes(
        load_model(config, arguments.earlier_checkpoint, device),
        dataset,
        label_map,
        device,
        config["initial_training"]["batch_size"],
    )
    second = prototypes(
        load_model(config, arguments.later_checkpoint, device),
        dataset,
        label_map,
        device,
        config["initial_training"]["batch_size"],
    )
    similarities = (first @ second.transpose(0, 1)).cpu().numpy()
    diagonal = np.diag(similarities)
    off_diagonal = similarities[~np.eye(similarities.shape[0], dtype=bool)]
    report = {
        "same_id_similarity_x100": 100.0 * float(diagonal.mean()),
        "different_id_similarity_x100": 100.0 * float(off_diagonal.mean()),
        "similarity_margin_x100": 100.0 * float(diagonal.mean() - off_diagonal.mean()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
