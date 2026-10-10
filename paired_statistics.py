import argparse
import json
from pathlib import Path

from pgrcl.metrics import holm_adjust, paired_improvement


def load_results(root: Path, method: str, seeds):
    return [
        json.loads(
            (root / method / "seed_{}".format(seed) / "results.json").read_text()
        )
        for seed in seeds
    ]


def condition_reports(root: Path, condition: str, seeds):
    pgrcl = load_results(root, "pgrcl", seeds)
    replay = load_results(root, "replay", seeds)
    metrics = [
        (
            "mean_historical_stage_accuracy",
            [item["mean_historical_stage_accuracy"] for item in pgrcl],
            [item["mean_historical_stage_accuracy"] for item in replay],
            True,
        ),
        (
            "T4_accuracy",
            [item["accuracy_trajectory"]["T4"]["T4"] for item in pgrcl],
            [item["accuracy_trajectory"]["T4"]["T4"] for item in replay],
            True,
        ),
        (
            "forgetting",
            [item["forgetting"] for item in pgrcl],
            [item["forgetting"] for item in replay],
            False,
        ),
    ]
    reports = []
    for name, first, second, higher in metrics:
        report = paired_improvement(first, second, higher)
        report["condition"] = condition
        report["metric"] = name
        reports.append(report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed-matched PGR-CL versus Replay statistics"
    )
    parser.add_argument("--full-results-root", default="outputs/cgsfnet_pgrcl")
    parser.add_argument("--controlled-results-root")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    arguments = parser.parse_args()
    reports = condition_reports(
        Path(arguments.full_results_root), "full_test_set", arguments.seeds
    )
    if arguments.controlled_results_root:
        reports.extend(
            condition_reports(
                Path(arguments.controlled_results_root),
                "controlled_subset",
                arguments.seeds,
            )
        )
    adjusted = holm_adjust([report["p_value"] for report in reports])
    for report, value in zip(reports, adjusted):
        report["holm_adjusted_p"] = value
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
