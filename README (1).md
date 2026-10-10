# CGSFNet and PGR-CL

Official implementation of **CGSFNet** and **Prototype-Guided Relation-Preserving Continual Learning (PGR-CL)** for fixed-identity continual sheep face identification across four growth stages.

The repository contains the improved backbone, the continual-learning method, all baselines used in the continual-learning comparison, the principal ablations, metric implementations, dataset leakage checks, and scripts for the five matched random seeds. The default configuration mirrors the experimental settings reported in the manuscript.

## Requirements

- Ubuntu 20.04
- Python 3.8
- PyTorch 2.0.1
- CUDA 11.8 and cuDNN 8.6 for GPU training

Create an environment and install the package from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
```

The code also runs on CPU, but the complete five-seed experiments are intended for GPU execution.

## Data

The dataset is not included because release is restricted by the participating farm's data-sharing agreement. See [data/README.md](data/README.md) for the manifest schema and integrity rules.

Place images under `data/images`, create `data/manifest.csv`, and audit the split before training:

```bash
cgsfnet-pgrcl audit --config configs/manuscript.yaml
```

The audit checks the stage labels, the common 180-identity class space, the reported 70/10/20 images per identity, image paths, and acquisition-group separation across training, validation, and test sets.

## ImageNet-1K initialization

The manuscript configuration requires an ImageNet-1K-pretrained LSNet-T checkpoint for structurally compatible layers. Install the optional dependency and prepare the checkpoint once:

```bash
python -m pip install -e '.[pretrained]'
python scripts/prepare_lsnet_checkpoint.py
```

The loader transfers only parameters with matching names and shapes. AdaLSConv, the 512-dimensional identity projection, and the 180-class identity head retain Kaiming initialization. Training stops with an explicit error if the required checkpoint is absent.

## Main experiments

Run PGR-CL for one seed:

```bash
cgsfnet-pgrcl continual --config configs/manuscript.yaml --method pgrcl --seed 0 --device cuda
```

Run the five manuscript seeds:

```bash
python scripts/run_all_seeds.py --method pgrcl --device cuda
```

The supported continual-learning methods are `finetune`, `ewc`, `lwf`, `replay`, and `pgrcl`. Each method uses the same CGSFNet backbone and T1→T2→T3→T4 arrival sequence. The offline reference is run separately:

```bash
cgsfnet-pgrcl joint --config configs/manuscript.yaml --seed 0 --device cuda
```

For the 4×4 fixed-model cross-stage matrix:

```bash
python scripts/run_cross_stage.py --device cuda
```

## Ablations and analyses

```bash
python scripts/run_backbone_ablations.py --device cuda
python scripts/run_sensitivity.py --device cuda
python scripts/storage_report.py
```

The backbone script evaluates the four contextual-kernel combinations, single-scale branches, fixed fusion, adaptive fusion, and the Stage 4 AdaLSBlock replacement. The sensitivity script varies one PGR-CL loss weight at a time over 0.1, 0.5, 1.0, and 2.0 while holding the other settings fixed.

After PGR-CL and Replay results are available for seeds 0–4, calculate the paired statistics for the full test set with:

```bash
python scripts/paired_statistics.py
```

For a test subset defined solely by acquisition metadata, evaluate the saved checkpoints and include all six comparisons in one Holm correction:

```bash
python scripts/evaluate_subset.py --method replay --filter-column camera_role --filter-value primary_frontal --device cuda
python scripts/evaluate_subset.py --method pgrcl --filter-column camera_role --filter-value primary_frontal --device cuda
python scripts/paired_statistics.py --controlled-results-root outputs/cgsfnet_pgrcl_controlled
```

`camera_role` is an acquisition-metadata field; subset selection does not use model predictions.

Cross-version prototype compatibility for one checkpoint pair is calculated with:

```bash
python scripts/prototype_compatibility.py \
  --earlier-checkpoint outputs/cgsfnet_pgrcl/pgrcl/seed_0/T1_complete.pt \
  --later-checkpoint outputs/cgsfnet_pgrcl/pgrcl/seed_0/T2_complete.pt \
  --stage T1 --device cuda
```

## Reproducibility details

- Training augmentation: horizontal flip with probability 0.5, random rotation within ±30°, and brightness/contrast/saturation jitter of 0.2; hue is unchanged.
- Initial or independent training: AdamW, learning rate 5×10⁻⁴, weight decay 10⁻⁴, batch size 32, and 100 epochs.
- Continual updates: a newly initialized AdamW optimizer and cosine scheduler at each stage, learning rate 10⁻⁴, weight decay 10⁻⁴, batch size 32, and 50 epochs.
- Both schedules use `CosineAnnealingLR` with a minimum learning rate of 10⁻⁶, no warm-up, and no early stopping.
- The retained checkpoint has the highest current-stage validation Accuracy. Test sets are accessed only after checkpoint selection.
- All principal experiments use seeds 0, 1, 2, 3, and 4.

PGR-CL uses loss weights (1.0, 1.0, 0.5) for feature distillation, the historical-prototype constraint, and relation preservation. EWC uses a regularization weight of 100. LwF uses temperature 2 and distillation weight 1.0. Replay stores 10 RGB uint8 exemplars per identity and combines 16 current images with 16 replay images during T2–T4.

## Verification

Generate a small synthetic dataset and run the tests:

```bash
python scripts/make_synthetic_dataset.py
python -m pip install -e '.[test]'
pytest
```

Synthetic data are intended only for checking installation and execution. They are not used to reproduce manuscript results.

## Repository structure

```text
configs/              manuscript and smoke-test configurations
data/                 dataset schema documentation
scripts/              experiment and analysis entry points
src/pgrcl/            model, losses, memory, training, and metrics
tests/                 architecture, loss, memory, metric, and data tests
```

## License

This code is released under the MIT License. Dataset access is governed separately by the participating farm's data-sharing agreement.
