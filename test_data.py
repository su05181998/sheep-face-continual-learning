from pathlib import Path

import pandas as pd
from PIL import Image

from pgrcl.data import DatasetSpecification, read_manifest, validate_manifest


def test_manifest_audit_accepts_group_separated_data(tmp_path: Path):
    rows = []
    for stage in ("T1", "T2", "T3", "T4"):
        for identity in ("A", "B"):
            for split in ("train", "validation", "test"):
                filename = "{}_{}_{}.png".format(stage, identity, split)
                Image.new("RGB", (8, 8)).save(tmp_path / filename)
                rows.append(
                    {
                        "path": filename,
                        "identity": identity,
                        "stage": stage,
                        "split": split,
                        "group_id": filename,
                        "acquisition_type": "fixed",
                    }
                )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    report = validate_manifest(
        read_manifest(manifest),
        DatasetSpecification(
            root=tmp_path,
            stages=["T1", "T2", "T3", "T4"],
            num_classes=2,
            expected_per_identity={"train": 1, "validation": 1, "test": 1},
            strict_counts=True,
        ),
    )
    assert report["images"] == 24
    assert report["cross_subset_group_overlap"] == 0
