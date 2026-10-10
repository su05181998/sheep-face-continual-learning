# Dataset layout

The image data are not distributed with this repository. The dataset is subject to the participating farm's data-sharing agreement.

Create `data/manifest.csv` with one row per image and these columns:

| Column | Meaning |
| --- | --- |
| `path` | Image path relative to `data/images`, or an absolute path |
| `identity` | Stable sheep identity label shared across T1–T4 |
| `stage` | One of `T1`, `T2`, `T3`, or `T4` |
| `split` | `train`, `validation`, or `test` |
| `group_id` | Passage-event or handheld-sequence identifier |
| `acquisition_type` | Acquisition source recorded in the metadata |

Additional acquisition metadata, such as `camera_role`, may be included for controlled-subset evaluation. These fields must be derived from acquisition records rather than model predictions.

The manuscript configuration validates 180 identities at every stage and 70/10/20 images per identity in the training/validation/test sets. A `group_id` may occur in only one subset within a stage. Run the audit before training:

```bash
cgsfnet-pgrcl audit --config configs/manuscript.yaml
```

For a self-contained installation check, generate the small synthetic dataset used by `configs/smoke.yaml`:

```bash
python scripts/make_synthetic_dataset.py
```
