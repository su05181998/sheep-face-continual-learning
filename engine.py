import copy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .losses import (
    PGRCLLoss,
    classification_loss,
    ewc_penalty,
    logit_distillation_loss,
)
from .memory import PerIdentityReservoir, PrototypeMemory
from .metrics import classification_metrics
from .utils import atomic_torch_save, load_torch_file


def freeze_copy(model: nn.Module, device: torch.device) -> nn.Module:
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    num_classes: int,
    return_predictions: bool = False,
) -> Dict[str, object]:
    model.eval()
    targets: List[int] = []
    predictions: List[int] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    result: Dict[str, object] = classification_metrics(
        targets, predictions, num_classes
    )
    if return_predictions:
        result["targets"] = targets
        result["predictions"] = predictions
    return result


@torch.no_grad()
def construct_stage_prototypes(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    num_classes: int,
    embedding_dim: int,
) -> Tensor:
    model.eval()
    sums = torch.zeros(num_classes, embedding_dim, dtype=torch.float64, device=device)
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        features = model.forward_features(images, normalize=True).double()
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels))
    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].cpu().tolist()
        raise ValueError("No training images for identities: {}".format(missing))
    prototypes = sums / counts.unsqueeze(1)
    return F.normalize(prototypes.float(), p=2, dim=1).cpu()


def estimate_empirical_fisher(
    model: nn.Module, loader: Iterable, device: torch.device
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Compute the diagonal empirical Fisher from every current-stage image."""
    model.eval()
    named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    fisher = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in named_parameters.items()
    }
    sample_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        for image, label in zip(images, labels):
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(image.unsqueeze(0)), label.unsqueeze(0))
            loss.backward()
            for name, parameter in named_parameters.items():
                if parameter.grad is not None:
                    fisher[name].add_(parameter.grad.detach().pow(2))
            sample_count += 1
    if sample_count == 0:
        raise ValueError("Cannot estimate Fisher information from an empty dataset")
    fisher = {name: value.div(sample_count).cpu() for name, value in fisher.items()}
    reference = {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters.items()
    }
    model.zero_grad(set_to_none=True)
    return fisher, reference


def _move_tensor_dict(values: Optional[Mapping[str, Tensor]], device: torch.device):
    if values is None:
        return None
    return {name: value.to(device) for name, value in values.items()}


def train_stage(
    model: nn.Module,
    train_loader: Iterable,
    validation_loaders: Mapping[str, Iterable],
    checkpoint_metric_stage: Optional[str],
    device: torch.device,
    num_classes: int,
    training_config: Mapping[str, object],
    method: str,
    method_config: Mapping[str, object],
    checkpoint_path: Path,
    teacher: Optional[nn.Module] = None,
    prototype_memory: Optional[PrototypeMemory] = None,
    fisher_diagonal: Optional[Mapping[str, Tensor]] = None,
    reference_parameters: Optional[Mapping[str, Tensor]] = None,
    replay_buffer: Optional[PerIdentityReservoir] = None,
    replay_transform=None,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["t_max"]),
        eta_min=float(training_config["minimum_learning_rate"]),
    )
    pgrcl_loss = None
    historical_references = None
    if method == "pgrcl" and teacher is not None:
        if prototype_memory is None:
            raise ValueError("PGR-CL requires prototype memory after T1")
        pgrcl_loss = PGRCLLoss(
            method_config["lambda_feature_distillation"],
            method_config["lambda_historical_prototype"],
            method_config["lambda_relation_preservation"],
        )
        historical_references = prototype_memory.reference(
            str(method_config.get("prototype_reference", "historical")), device
        )
    fisher_device = _move_tensor_dict(fisher_diagonal, device)
    reference_device = _move_tensor_dict(reference_parameters, device)
    best_accuracy = -float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if method == "replay" and replay_buffer is not None and len(replay_buffer):
                replay_entries = replay_buffer.sample(
                    int(method_config["replay_batch_size"])
                )
                replay_images = torch.stack(
                    [replay_transform(entry.image) for entry in replay_entries]
                ).to(device, non_blocking=True)
                replay_labels = torch.tensor(
                    [entry.label for entry in replay_entries],
                    dtype=torch.long,
                    device=device,
                )
                images = torch.cat([images, replay_images], dim=0)
                labels = torch.cat([labels, replay_labels], dim=0)

            optimizer.zero_grad(set_to_none=True)
            logits, features = model(images, return_features=True)
            if method == "pgrcl" and teacher is not None:
                with torch.no_grad():
                    _, teacher_features = teacher(images, return_features=True)
                losses = pgrcl_loss(
                    logits,
                    features,
                    labels,
                    teacher_features,
                    historical_references,
                )
                loss = losses["total"]
            elif method == "lwf" and teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                kd = logit_distillation_loss(
                    logits, teacher_logits, float(method_config["temperature"])
                )
                loss = (
                    classification_loss(logits, labels)
                    + float(method_config["lambda_kd"]) * kd
                )
            else:
                loss = classification_loss(logits, labels)
                if method == "ewc" and fisher_device is not None:
                    loss = loss + float(method_config["lambda"]) * ewc_penalty(
                        model, reference_device, fisher_device
                    )
            loss.backward()
            optimizer.step()
            batch_size = labels.shape[0]
            running_loss += float(loss.detach()) * batch_size
            seen += batch_size
        scheduler.step()

        validation = {
            stage: evaluate(model, loader, device, num_classes)["accuracy"]
            for stage, loader in validation_loaders.items()
        }
        selection_accuracy = (
            float(validation[checkpoint_metric_stage])
            if checkpoint_metric_stage is not None
            else float(np.mean(list(validation.values())))
        )
        record = {
            "epoch": float(epoch),
            "loss": running_loss / max(1, seen),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "selection_accuracy": selection_accuracy,
        }
        for stage, value in validation.items():
            record["validation_{}_accuracy".format(stage)] = float(value)
        history.append(record)
        if selection_accuracy > best_accuracy:
            best_accuracy = selection_accuracy
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "selection_accuracy": selection_accuracy,
                },
                checkpoint_path,
            )

    best = load_torch_file(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    return history
