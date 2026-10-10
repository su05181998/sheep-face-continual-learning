import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class PrototypeMemory:
    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self._stages: Dict[str, Tensor] = {}

    @property
    def stages(self) -> List[str]:
        return list(self._stages.keys())

    def add(self, stage: str, prototypes: Tensor) -> None:
        expected = (self.num_classes, self.embedding_dim)
        if tuple(prototypes.shape) != expected:
            raise ValueError(
                "Expected prototype matrix {} but received {}".format(
                    expected, tuple(prototypes.shape)
                )
            )
        self._stages[str(stage)] = F.normalize(
            prototypes.detach().cpu().float(), p=2, dim=1
        )

    def reference(self, mode: str = "historical", device=None) -> Tensor:
        if not self._stages:
            raise RuntimeError("Prototype memory is empty")
        values = list(self._stages.values())
        if mode == "latest":
            reference = values[-1]
        elif mode == "historical":
            reference = F.normalize(torch.stack(values).mean(dim=0), p=2, dim=1)
        else:
            raise ValueError("Prototype reference must be historical or latest")
        return reference.to(device=device) if device is not None else reference

    def state_dict(self) -> Dict[str, object]:
        return {
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "stages": {key: value.clone() for key, value in self._stages.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["num_classes"]) != self.num_classes:
            raise ValueError("Prototype memory class count does not match")
        if int(state["embedding_dim"]) != self.embedding_dim:
            raise ValueError("Prototype memory embedding dimension does not match")
        self._stages = {
            str(key): value.detach().cpu().float().clone()
            for key, value in state["stages"].items()
        }


@dataclass
class ReplayExample:
    image: np.ndarray
    label: int


class PerIdentityReservoir:
    def __init__(self, capacity_per_identity: int, seed: int) -> None:
        self.capacity_per_identity = int(capacity_per_identity)
        if self.capacity_per_identity < 1:
            raise ValueError("Replay capacity per identity must be positive")
        self._rng = random.Random(seed)
        self._examples: DefaultDict[int, List[ReplayExample]] = defaultdict(list)
        self._seen: DefaultDict[int, int] = defaultdict(int)

    def update(self, samples: Sequence[Tuple[np.ndarray, int]]) -> None:
        for image, label_value in samples:
            label = int(label_value)
            array = np.asarray(image, dtype=np.uint8)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("Replay images must be H x W x 3 RGB uint8 arrays")
            self._seen[label] += 1
            bucket = self._examples[label]
            example = ReplayExample(array.copy(), label)
            if len(bucket) < self.capacity_per_identity:
                bucket.append(example)
            else:
                index = self._rng.randrange(self._seen[label])
                if index < self.capacity_per_identity:
                    bucket[index] = example

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._examples.values())

    def examples(self) -> List[ReplayExample]:
        result: List[ReplayExample] = []
        for label in sorted(self._examples):
            result.extend(self._examples[label])
        return result

    def sample(self, count: int) -> List[ReplayExample]:
        population = self.examples()
        if not population:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        if count <= len(population):
            return self._rng.sample(population, count)
        return [self._rng.choice(population) for _ in range(count)]

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity_per_identity": self.capacity_per_identity,
            "seen": dict(self._seen),
            "examples": {
                int(label): [(entry.image, entry.label) for entry in bucket]
                for label, bucket in self._examples.items()
            },
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["capacity_per_identity"]) != self.capacity_per_identity:
            raise ValueError("Replay capacity does not match")
        self._seen = defaultdict(
            int, {int(k): int(v) for k, v in state["seen"].items()}
        )
        self._examples = defaultdict(list)
        for label, entries in state["examples"].items():
            self._examples[int(label)] = [
                ReplayExample(np.asarray(image, dtype=np.uint8).copy(), int(target))
                for image, target in entries
            ]
        self._rng.setstate(state["rng_state"])
