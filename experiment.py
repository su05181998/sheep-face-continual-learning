from typing import Dict, Mapping, Optional, Sequence

import torch

from .config import resolve_path
from .data import (
    DatasetSpecification,
    SheepFaceDataset,
    build_transforms,
    identity_mapping,
    make_loader,
    read_manifest,
    select_rows,
    validate_manifest,
)
from .engine import (
    construct_stage_prototypes,
    estimate_empirical_fisher,
    evaluate,
    freeze_copy,
    train_stage,
)
from .memory import PerIdentityReservoir, PrototypeMemory
from .metrics import average_forgetting
from .model import build_cgsfnet
from .utils import save_json, seed_everything, select_device

SUPPORTED_METHODS = {"finetune", "ewc", "lwf", "replay", "pgrcl"}


class Experiment:
    def __init__(self, config: Dict[str, object], device: str = "auto") -> None:
        self.config = config
        self.device = select_device(device)
        self.stages: Sequence[str] = config["experiment"]["stages"]
        data_config = config["data"]
        self.data_root = resolve_path(config, data_config["root"])
        self.manifest_path = resolve_path(config, data_config["manifest"])
        self.frame = read_manifest(self.manifest_path)
        self.specification = DatasetSpecification(
            root=self.data_root,
            stages=self.stages,
            num_classes=int(data_config["num_classes"]),
            expected_per_identity=data_config["expected_per_identity"],
            strict_counts=bool(data_config["strict_counts"]),
        )
        self.audit = validate_manifest(self.frame, self.specification)
        self.label_map = identity_mapping(self.frame)
        self.train_transform, self.eval_transform, self.replay_transform = (
            build_transforms(
                int(data_config["image_size"]),
                data_config["mean"],
                data_config["std"],
                config["augmentation"],
            )
        )
        self.output_root = resolve_path(config, config["experiment"]["output_dir"])

    def _dataset(self, stage: Optional[str], split: str, training: bool):
        return SheepFaceDataset(
            select_rows(self.frame, stage, split),
            self.data_root,
            self.label_map,
            self.train_transform if training else self.eval_transform,
            int(self.config["data"]["image_size"]),
        )

    def _loader(
        self,
        stage: Optional[str],
        split: str,
        batch_size: int,
        seed: int,
        training: bool,
    ):
        return make_loader(
            self._dataset(stage, split, training),
            batch_size=batch_size,
            shuffle=training,
            workers=int(self.config["data"]["workers"]),
            pin_memory=bool(self.config["data"]["pin_memory"]),
            seed=seed,
            drop_last=training,
        )

    def _new_model(self) -> torch.nn.Module:
        model = build_cgsfnet(self.config["model"])
        checkpoint = resolve_path(
            self.config, self.config["model"].get("imagenet_checkpoint")
        )
        require_checkpoint = bool(
            self.config["model"].get("require_imagenet_checkpoint", False)
        )
        if checkpoint is not None and checkpoint.is_file():
            report = model.load_compatible_lsnet_t_weights(checkpoint)
            if not report["loaded"]:
                raise RuntimeError("The ImageNet checkpoint has no compatible weights")
        elif require_checkpoint:
            raise FileNotFoundError(
                "Required ImageNet-1K LSNet-T checkpoint not found: {}".format(
                    checkpoint
                )
            )
        return model.to(self.device)

    def run_continual(self, method: str, seed: int) -> Dict[str, object]:
        method = method.lower()
        if method not in SUPPORTED_METHODS:
            raise ValueError("Unsupported continual-learning method: {}".format(method))
        seed_everything(seed)
        run_dir = self.output_root / method / "seed_{}".format(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        model = self._new_model()
        prototypes = (
            PrototypeMemory(
                int(self.config["model"]["num_classes"]),
                int(self.config["model"]["embedding_dim"]),
            )
            if method == "pgrcl"
            else None
        )
        replay_buffer = PerIdentityReservoir(
            int(self.config["replay"]["capacity_per_identity"]), seed
        )
        fisher = None
        reference = None
        trajectory: Dict[str, Dict[str, float]] = {}
        stage_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

        for stage_index, stage in enumerate(self.stages):
            teacher = (
                freeze_copy(model, self.device)
                if stage_index > 0 and method in {"lwf", "pgrcl"}
                else None
            )
            training_config = (
                self.config["initial_training"]
                if stage_index == 0
                else self.config["continual_training"]
            )
            if method == "replay" and stage_index > 0:
                train_batch_size = int(self.config["replay"]["current_batch_size"])
            else:
                train_batch_size = int(training_config["batch_size"])
            train_loader = self._loader(
                stage, "train", train_batch_size, seed + stage_index, True
            )
            validation_loader = self._loader(
                stage,
                "validation",
                int(training_config["batch_size"]),
                seed,
                False,
            )
            if stage_index == 0:
                stage_method = "finetune"
                method_config: Mapping[str, object] = {}
            else:
                stage_method = method
                method_config = self.config.get(method, {})
            history = train_stage(
                model=model,
                train_loader=train_loader,
                validation_loaders={stage: validation_loader},
                checkpoint_metric_stage=stage,
                device=self.device,
                num_classes=int(self.config["data"]["num_classes"]),
                training_config=training_config,
                method=stage_method,
                method_config=method_config,
                checkpoint_path=run_dir / "{}_best.pt".format(stage),
                teacher=teacher,
                prototype_memory=prototypes,
                fisher_diagonal=fisher,
                reference_parameters=reference,
                replay_buffer=replay_buffer,
                replay_transform=self.replay_transform,
            )
            save_json(history, run_dir / "{}_history.json".format(stage))

            prototype_loader = self._loader(
                stage,
                "train",
                int(training_config["batch_size"]),
                seed,
                False,
            )
            if method == "pgrcl":
                prototypes.add(
                    stage,
                    construct_stage_prototypes(
                        model,
                        prototype_loader,
                        self.device,
                        int(self.config["model"]["num_classes"]),
                        int(self.config["model"]["embedding_dim"]),
                    ),
                )
            if method == "ewc":
                fisher, reference = estimate_empirical_fisher(
                    model, prototype_loader, self.device
                )
            if method == "replay":
                dataset = self._dataset(stage, "train", training=False)
                replay_buffer.update(
                    [
                        (
                            dataset.raw_resized_uint8(index),
                            self.label_map[str(dataset.frame.iloc[index]["identity"])],
                        )
                        for index in range(len(dataset))
                    ]
                )
                expected_capacity = int(self.config["replay"]["total_capacity"])
                if len(replay_buffer) > expected_capacity:
                    raise RuntimeError("Replay buffer exceeds its configured capacity")

            stage_results: Dict[str, Dict[str, float]] = {}
            for test_stage in self.stages[: stage_index + 1]:
                loader = self._loader(
                    test_stage,
                    "test",
                    int(training_config["batch_size"]),
                    seed,
                    False,
                )
                stage_results[test_stage] = evaluate(
                    model,
                    loader,
                    self.device,
                    int(self.config["data"]["num_classes"]),
                )
            stage_metrics[stage] = stage_results
            trajectory[stage] = {
                key: float(value["accuracy"]) for key, value in stage_results.items()
            }
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "prototype_memory": prototypes.state_dict()
                    if method == "pgrcl"
                    else None,
                    "replay_buffer": replay_buffer.state_dict()
                    if method == "replay"
                    else None,
                    "stage": stage,
                    "seed": seed,
                    "method": method,
                    "label_map": self.label_map,
                },
                run_dir / "{}_complete.pt".format(stage),
            )

        result: Dict[str, object] = {
            "method": method,
            "seed": seed,
            "stage_metrics": stage_metrics,
            "accuracy_trajectory": trajectory,
            "forgetting": average_forgetting(trajectory, self.stages),
            "mean_historical_stage_accuracy": sum(
                trajectory[self.stages[-1]][stage] for stage in self.stages[:-1]
            )
            / float(len(self.stages) - 1),
        }
        save_json(result, run_dir / "results.json")
        return result

    def run_independent_stage(self, source_stage: str, seed: int) -> Dict[str, object]:
        if source_stage not in self.stages:
            raise ValueError("Unknown source stage: {}".format(source_stage))
        seed_everything(seed)
        run_dir = (
            self.output_root / "cross_stage" / source_stage / "seed_{}".format(seed)
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        model = self._new_model()
        config = self.config["initial_training"]
        history = train_stage(
            model=model,
            train_loader=self._loader(
                source_stage, "train", int(config["batch_size"]), seed, True
            ),
            validation_loaders={
                source_stage: self._loader(
                    source_stage,
                    "validation",
                    int(config["batch_size"]),
                    seed,
                    False,
                )
            },
            checkpoint_metric_stage=source_stage,
            device=self.device,
            num_classes=int(self.config["data"]["num_classes"]),
            training_config=config,
            method="finetune",
            method_config={},
            checkpoint_path=run_dir / "best.pt",
        )
        results = {}
        for test_stage in self.stages:
            results[test_stage] = evaluate(
                model,
                self._loader(
                    test_stage,
                    "test",
                    int(config["batch_size"]),
                    seed,
                    False,
                ),
                self.device,
                int(self.config["data"]["num_classes"]),
            )
        payload = {"source_stage": source_stage, "seed": seed, "test_metrics": results}
        save_json(history, run_dir / "history.json")
        save_json(payload, run_dir / "results.json")
        return payload

    def run_joint(self, seed: int) -> Dict[str, object]:
        seed_everything(seed)
        run_dir = self.output_root / "joint" / "seed_{}".format(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        model = self._new_model()
        config = self.config["initial_training"]
        validation_loaders = {
            stage: self._loader(
                stage, "validation", int(config["batch_size"]), seed, False
            )
            for stage in self.stages
        }
        history = train_stage(
            model=model,
            train_loader=self._loader(
                None, "train", int(config["batch_size"]), seed, True
            ),
            validation_loaders=validation_loaders,
            checkpoint_metric_stage=None,
            device=self.device,
            num_classes=int(self.config["data"]["num_classes"]),
            training_config=config,
            method="finetune",
            method_config={},
            checkpoint_path=run_dir / "best.pt",
        )
        results = {
            stage: evaluate(
                model,
                self._loader(stage, "test", int(config["batch_size"]), seed, False),
                self.device,
                int(self.config["data"]["num_classes"]),
            )
            for stage in self.stages
        }
        payload = {"seed": seed, "test_metrics": results}
        save_json(history, run_dir / "history.json")
        save_json(payload, run_dir / "results.json")
        return payload
