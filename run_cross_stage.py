import argparse

from pgrcl.config import load_config
from pgrcl.experiment import Experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 4 x 4 CGSFNet evaluation")
    parser.add_argument("--config", default="configs/manuscript.yaml")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    experiment = Experiment(config, arguments.device)
    for stage in config["experiment"]["stages"]:
        for seed in config["experiment"]["seeds"]:
            experiment.run_independent_stage(stage, int(seed))


if __name__ == "__main__":
    main()
