import argparse
import copy
from pathlib import Path

from pgrcl.config import load_config
from pgrcl.experiment import Experiment

KEYS = {
    "lambda1": "lambda_feature_distillation",
    "lambda2": "lambda_historical_prototype",
    "lambda3": "lambda_relation_preservation",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one-factor-at-a-time loss sensitivity"
    )
    parser.add_argument("--config", default="configs/manuscript.yaml")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    base = load_config(arguments.config)
    for short_name, key in KEYS.items():
        for value in (0.1, 0.5, 1.0, 2.0):
            config = copy.deepcopy(base)
            config["pgrcl"][key] = value
            root = Path(base["experiment"]["output_dir"])
            config["experiment"]["output_dir"] = str(
                root / "sensitivity" / short_name / str(value)
            )
            experiment = Experiment(config, arguments.device)
            for seed in config["experiment"]["seeds"]:
                experiment.run_continual("pgrcl", int(seed))


if __name__ == "__main__":
    main()
