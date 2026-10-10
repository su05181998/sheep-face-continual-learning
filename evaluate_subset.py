import argparse
from pathlib import Path

from pgrcl.config import load_config
from pgrcl.engine import evaluate
from pgrcl.experiment import Experiment
from pgrcl.metrics import average_forgetting
from pgrcl.model import build_cgsfnet
from pgrcl.utils import load_torch_file, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved stage checkpoints on a metadata-defined test subset"
    )
    parser.add_argument("--config", default="configs/manuscript.yaml")
    parser.add_argument("--method", choices=["replay", "pgrcl"], required=True)
    parser.add_argument("--filter-column", required=True)
    parser.add_argument("--filter-value", required=True)
    parser.add_argument("--input-root", default="outputs/cgsfnet_pgrcl")
    parser.add_argument("--output-root", default="outputs/cgsfnet_pgrcl_controlled")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    experiment = Experiment(config, arguments.device)
    if arguments.filter_column not in experiment.frame.columns:
        raise ValueError(
            "Manifest column not found: {}".format(arguments.filter_column)
        )
    retained = (
        experiment.frame[arguments.filter_column].astype(str).eq(arguments.filter_value)
    )
    experiment.frame = experiment.frame[
        experiment.frame["split"].ne("test") | retained
    ].copy()
    for stage in experiment.stages:
        count = len(
            experiment.frame[
                experiment.frame["stage"].eq(stage)
                & experiment.frame["split"].eq("test")
            ]
        )
        if count == 0:
            raise ValueError("The filtered test subset is empty at {}".format(stage))

    for seed in config["experiment"]["seeds"]:
        trajectory = {}
        stage_metrics = {}
        for stage_index, model_stage in enumerate(experiment.stages):
            checkpoint = (
                Path(arguments.input_root)
                / arguments.method
                / "seed_{}".format(seed)
                / "{}_complete.pt".format(model_stage)
            )
            payload = load_torch_file(
                checkpoint, map_location=experiment.device, weights_only=False
            )
            model = build_cgsfnet(config["model"]).to(experiment.device)
            model.load_state_dict(payload["model_state"])
            results = {}
            for test_stage in experiment.stages[: stage_index + 1]:
                results[test_stage] = evaluate(
                    model,
                    experiment._loader(
                        test_stage,
                        "test",
                        int(config["continual_training"]["batch_size"]),
                        int(seed),
                        False,
                    ),
                    experiment.device,
                    int(config["data"]["num_classes"]),
                )
            stage_metrics[model_stage] = results
            trajectory[model_stage] = {
                key: float(value["accuracy"]) for key, value in results.items()
            }
        result = {
            "method": arguments.method,
            "seed": int(seed),
            "filter": {
                "column": arguments.filter_column,
                "value": arguments.filter_value,
            },
            "stage_metrics": stage_metrics,
            "accuracy_trajectory": trajectory,
            "forgetting": average_forgetting(trajectory, experiment.stages),
            "mean_historical_stage_accuracy": sum(
                trajectory[experiment.stages[-1]][stage]
                for stage in experiment.stages[:-1]
            )
            / float(len(experiment.stages) - 1),
        }
        output = (
            Path(arguments.output_root)
            / arguments.method
            / "seed_{}".format(seed)
            / "results.json"
        )
        save_json(result, output)


if __name__ == "__main__":
    main()
