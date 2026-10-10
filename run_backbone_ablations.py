import argparse
import copy
from pathlib import Path

from pgrcl.config import load_config
from pgrcl.experiment import Experiment

VARIANTS = {
    "7x7_only": {
        "small_context_kernel": 7,
        "large_context_kernel": 11,
        "context_mode": "small",
    },
    "11x11_only": {
        "small_context_kernel": 7,
        "large_context_kernel": 11,
        "context_mode": "large",
    },
    "7x7_11x11_fixed": {
        "small_context_kernel": 7,
        "large_context_kernel": 11,
        "context_mode": "fixed",
    },
    "5x5_9x9": {
        "small_context_kernel": 5,
        "large_context_kernel": 9,
        "context_mode": "adaptive",
    },
    "5x5_11x11": {
        "small_context_kernel": 5,
        "large_context_kernel": 11,
        "context_mode": "adaptive",
    },
    "7x7_9x9": {
        "small_context_kernel": 7,
        "large_context_kernel": 9,
        "context_mode": "adaptive",
    },
    "7x7_11x11_stage4_adals": {
        "small_context_kernel": 7,
        "large_context_kernel": 11,
        "context_mode": "adaptive",
        "stage4_block": "adals",
    },
    "complete_cgsfnet": {
        "small_context_kernel": 7,
        "large_context_kernel": 11,
        "context_mode": "adaptive",
        "stage4_block": "msa",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CGSFNet component ablations")
    parser.add_argument("--config", default="configs/manuscript.yaml")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    base = load_config(arguments.config)
    for name, changes in VARIANTS.items():
        config = copy.deepcopy(base)
        config["model"].update(changes)
        root = Path(base["experiment"]["output_dir"])
        config["experiment"]["output_dir"] = str(root / "backbone_ablation" / name)
        experiment = Experiment(config, arguments.device)
        for stage in config["experiment"]["stages"]:
            for seed in config["experiment"]["seeds"]:
                experiment.run_independent_stage(stage, int(seed))


if __name__ == "__main__":
    main()
