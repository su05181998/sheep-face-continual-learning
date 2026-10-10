import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config
from .data import DatasetSpecification, read_manifest, validate_manifest
from .experiment import SUPPORTED_METHODS, Experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cgsfnet-pgrcl",
        description="Train and evaluate CGSFNet and PGR-CL.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate a dataset manifest")
    audit.add_argument("--config", default="configs/manuscript.yaml")

    continual = subparsers.add_parser(
        "continual", help="Run one continual-learning method"
    )
    continual.add_argument("--config", default="configs/manuscript.yaml")
    continual.add_argument("--method", choices=sorted(SUPPORTED_METHODS), required=True)
    continual.add_argument("--seed", type=int, required=True)
    continual.add_argument("--device", default="auto")

    cross_stage = subparsers.add_parser(
        "cross-stage", help="Train on one stage and test on all stages"
    )
    cross_stage.add_argument("--config", default="configs/manuscript.yaml")
    cross_stage.add_argument(
        "--source-stage", choices=["T1", "T2", "T3", "T4"], required=True
    )
    cross_stage.add_argument("--seed", type=int, required=True)
    cross_stage.add_argument("--device", default="auto")

    joint = subparsers.add_parser(
        "joint", help="Run the offline Joint Training reference"
    )
    joint.add_argument("--config", default="configs/manuscript.yaml")
    joint.add_argument("--seed", type=int, required=True)
    joint.add_argument("--device", default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = _parser().parse_args(argv)
    config = load_config(arguments.config)
    if arguments.command == "audit":
        data = config["data"]
        root = Path(config["_project_root"]) / data["root"]
        manifest = Path(config["_project_root"]) / data["manifest"]
        report = validate_manifest(
            read_manifest(manifest),
            DatasetSpecification(
                root=root,
                stages=config["experiment"]["stages"],
                num_classes=int(data["num_classes"]),
                expected_per_identity=data["expected_per_identity"],
                strict_counts=bool(data["strict_counts"]),
            ),
        )
        print(json.dumps(report, indent=2))
        return
    experiment = Experiment(config, arguments.device)
    if arguments.command == "continual":
        result = experiment.run_continual(arguments.method, arguments.seed)
    elif arguments.command == "cross-stage":
        result = experiment.run_independent_stage(
            arguments.source_stage, arguments.seed
        )
    else:
        result = experiment.run_joint(arguments.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
