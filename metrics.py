from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> Dict[str, float]:
    targets_array = np.asarray(targets, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    if targets_array.shape != predictions_array.shape or targets_array.size == 0:
        raise ValueError("Targets and predictions must be non-empty and equally sized")
    labels = np.arange(num_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_array,
        predictions_array,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    accuracy = (targets_array == predictions_array).mean()
    return {
        "accuracy": 100.0 * float(accuracy),
        "precision": 100.0 * float(precision),
        "recall": 100.0 * float(recall),
        "f1": 100.0 * float(f1),
    }


def acpv(stage_metrics: Sequence[Mapping[str, float]]) -> float:
    if len(stage_metrics) != 4:
        raise ValueError("ACPV requires four independently trained same-stage results")
    ranges = []
    for key in ("accuracy", "precision", "f1"):
        values = [float(metrics[key]) for metrics in stage_metrics]
        ranges.append(max(values) - min(values))
    return float(np.mean(ranges))


def average_forgetting(
    accuracy_trajectory: Mapping[str, Mapping[str, float]],
    stages: Sequence[str],
) -> float:
    final_stage = stages[-1]
    if final_stage not in accuracy_trajectory:
        raise ValueError("The final-stage evaluation is missing")
    losses = []
    for historical_index, historical_stage in enumerate(stages[:-1]):
        values = []
        for model_stage in stages[historical_index:]:
            if model_stage in accuracy_trajectory:
                stage_results = accuracy_trajectory[model_stage]
                if historical_stage in stage_results:
                    values.append(float(stage_results[historical_stage]))
        if not values or historical_stage not in accuracy_trajectory[final_stage]:
            raise ValueError("Incomplete accuracy trajectory for Forgetting")
        losses.append(
            max(values) - float(accuracy_trajectory[final_stage][historical_stage])
        )
    return float(np.mean(losses))


def individual_accuracy(
    targets: Sequence[int], predictions: Sequence[int]
) -> Dict[int, float]:
    targets_array = np.asarray(targets, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    result = {}
    for identity in np.unique(targets_array):
        mask = targets_array == identity
        result[int(identity)] = 100.0 * float(
            (predictions_array[mask] == targets_array[mask]).mean()
        )
    return result


def paired_improvement(
    pgrcl: Iterable[float], replay: Iterable[float], higher_is_better: bool = True
) -> Dict[str, float]:
    first = np.asarray(list(pgrcl), dtype=np.float64)
    second = np.asarray(list(replay), dtype=np.float64)
    if first.shape != second.shape or first.size < 2:
        raise ValueError(
            "Paired samples must have equal length and at least two values"
        )
    differences = first - second if higher_is_better else second - first
    mean_difference = float(differences.mean())
    standard_deviation = float(differences.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(differences.size)
    critical = stats.t.ppf(0.975, differences.size - 1)
    _, p_value = stats.ttest_1samp(differences, 0.0)
    return {
        "mean_difference": mean_difference,
        "ci_low": mean_difference - critical * standard_error,
        "ci_high": mean_difference + critical * standard_error,
        "p_value": float(p_value),
        "cohens_dz": mean_difference / standard_deviation,
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (len(values) - np.arange(len(values))) * values[order])
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()
