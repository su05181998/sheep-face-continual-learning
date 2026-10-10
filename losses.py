from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits, labels)


def feature_distillation_loss(student: Tensor, teacher: Tensor) -> Tensor:
    student = F.normalize(student, p=2, dim=1)
    teacher = F.normalize(teacher.detach(), p=2, dim=1)
    return (1.0 - (student * teacher).sum(dim=1)).mean()


def historical_prototype_loss(
    student: Tensor, labels: Tensor, historical_references: Tensor
) -> Tensor:
    student = F.normalize(student, p=2, dim=1)
    references = F.normalize(historical_references[labels], p=2, dim=1)
    return (1.0 - (student * references).sum(dim=1)).mean()


def relation_preservation_loss(student: Tensor, teacher: Tensor) -> Tensor:
    student = F.normalize(student, p=2, dim=1)
    teacher = F.normalize(teacher.detach(), p=2, dim=1)
    student_relation = student @ student.transpose(0, 1)
    teacher_relation = teacher @ teacher.transpose(0, 1)
    batch_size = student.shape[0]
    return (student_relation - teacher_relation).pow(2).sum() / float(batch_size**2)


def logit_distillation_loss(
    student_logits: Tensor, teacher_logits: Tensor, temperature: float
) -> Tensor:
    scaled_student = F.log_softmax(student_logits / temperature, dim=1)
    scaled_teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return (temperature**2) * F.kl_div(
        scaled_student, scaled_teacher, reduction="batchmean"
    )


class PGRCLLoss(nn.Module):
    def __init__(
        self,
        lambda_feature_distillation: float,
        lambda_historical_prototype: float,
        lambda_relation_preservation: float,
    ) -> None:
        super().__init__()
        self.lambda_feature_distillation = float(lambda_feature_distillation)
        self.lambda_historical_prototype = float(lambda_historical_prototype)
        self.lambda_relation_preservation = float(lambda_relation_preservation)

    def forward(
        self,
        student_logits: Tensor,
        student_features: Tensor,
        labels: Tensor,
        teacher_features: Optional[Tensor] = None,
        historical_references: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {
            "classification": classification_loss(student_logits, labels)
        }
        if teacher_features is None or historical_references is None:
            losses["total"] = losses["classification"]
            return losses
        losses["feature_distillation"] = feature_distillation_loss(
            student_features, teacher_features
        )
        losses["historical_prototype"] = historical_prototype_loss(
            student_features, labels, historical_references
        )
        losses["relation_preservation"] = relation_preservation_loss(
            student_features, teacher_features
        )
        losses["total"] = (
            losses["classification"]
            + self.lambda_feature_distillation * losses["feature_distillation"]
            + self.lambda_historical_prototype * losses["historical_prototype"]
            + self.lambda_relation_preservation * losses["relation_preservation"]
        )
        return losses


def ewc_penalty(
    model: nn.Module,
    reference_parameters: Dict[str, Tensor],
    fisher_diagonal: Dict[str, Tensor],
) -> Tensor:
    trainable = dict(model.named_parameters())
    if not trainable:
        raise ValueError("The model has no trainable parameters")
    first = next(iter(trainable.values()))
    penalty = torch.zeros((), device=first.device)
    for name, parameter in trainable.items():
        if name in reference_parameters and name in fisher_diagonal:
            penalty = (
                penalty
                + (
                    fisher_diagonal[name]
                    * (parameter - reference_parameters[name]).pow(2)
                ).sum()
            )
    return penalty
