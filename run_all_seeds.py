import argparse

from pgrcl.config import load_config
from pgrcl.experiment import SUPPORTED_METHODS, Experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five manuscript seeds")
    parser.add_argument("--config", default="configs/manuscript.yaml")
    parser.add_argument("--method", choices=sorted(SUPPORTED_METHODS), required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    experiment = Experiment(config, arguments.device)
    for seed in config["experiment"]["seeds"]:
        experiment.run_continual(arguments.method, int(seed))


if __name__ == "__main__":
    main()
