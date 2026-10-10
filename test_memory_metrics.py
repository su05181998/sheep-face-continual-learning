import numpy as np
import torch
import torch.nn.functional as F

from pgrcl.memory import PerIdentityReservoir, PrototypeMemory
from pgrcl.metrics import acpv, average_forgetting, classification_metrics, holm_adjust


def test_prototype_memory_historical_reference():
    memory = PrototypeMemory(2, 3)
    first = F.normalize(torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]), dim=1)
    second = F.normalize(torch.tensor([[0.8, 0.2, 0], [0.2, 0.8, 0]]), dim=1)
    memory.add("T1", first)
    memory.add("T2", second)
    expected = F.normalize(torch.stack([first, second]).mean(dim=0), dim=1)
    assert torch.allclose(memory.reference("historical"), expected)
    assert torch.allclose(memory.reference("latest"), second)


def test_per_identity_reservoir_respects_capacity():
    buffer = PerIdentityReservoir(2, seed=0)
    samples = []
    for label in range(3):
        for value in range(10):
            samples.append((np.full((8, 8, 3), value, dtype=np.uint8), label))
    buffer.update(samples)
    assert len(buffer) == 6
    assert all(len(buffer._examples[label]) == 2 for label in range(3))


def test_metrics_follow_manuscript_definitions():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert metrics["accuracy"] == 75.0
    stages = [
        {"accuracy": 90, "precision": 89, "f1": 88},
        {"accuracy": 92, "precision": 91, "f1": 90},
        {"accuracy": 94, "precision": 93, "f1": 92},
        {"accuracy": 96, "precision": 95, "f1": 94},
    ]
    assert acpv(stages) == 6.0
    trajectory = {
        "T1": {"T1": 90},
        "T2": {"T1": 88, "T2": 92},
        "T3": {"T1": 87, "T2": 90, "T3": 94},
        "T4": {"T1": 85, "T2": 89, "T3": 93, "T4": 96},
    }
    assert average_forgetting(trajectory, ["T1", "T2", "T3", "T4"]) == 3.0
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
