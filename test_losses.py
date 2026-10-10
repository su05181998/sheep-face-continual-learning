import torch
import torch.nn.functional as F

from pgrcl.losses import (
    feature_distillation_loss,
    historical_prototype_loss,
    logit_distillation_loss,
    relation_preservation_loss,
)


def test_matching_features_have_zero_distillation_and_relation_loss():
    features = F.normalize(torch.randn(5, 16), dim=1)
    assert torch.allclose(
        feature_distillation_loss(features, features), torch.tensor(0.0), atol=1e-6
    )
    assert torch.allclose(
        relation_preservation_loss(features, features), torch.tensor(0.0), atol=1e-6
    )


def test_matching_prototypes_have_zero_loss():
    references = F.normalize(torch.randn(4, 16), dim=1)
    labels = torch.tensor([0, 2, 3])
    assert torch.allclose(
        historical_prototype_loss(references[labels], labels, references),
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_matching_logits_have_zero_kl():
    logits = torch.randn(5, 4)
    assert abs(float(logit_distillation_loss(logits, logits, 2.0))) < 1e-6
