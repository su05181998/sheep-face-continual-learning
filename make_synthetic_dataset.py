import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small deterministic test dataset"
    )
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-per-class", type=int, default=4)
    parser.add_argument("--validation-per-class", type=int, default=2)
    parser.add_argument("--test-per-class", type=int, default=2)
    arguments = parser.parse_args()
    output = Path(arguments.output)
    image_root = output / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": arguments.train_per_class,
        "validation": arguments.validation_per_class,
        "test": arguments.test_per_class,
    }
    rows = []
    rng = np.random.default_rng(2026)
    for stage_index, stage in enumerate(("T1", "T2", "T3", "T4")):
        for identity in range(arguments.classes):
            for split, count in counts.items():
                for sample_index in range(count):
                    base = np.zeros(
                        (arguments.image_size, arguments.image_size, 3), dtype=np.uint8
                    )
                    base[..., 0] = 35 + identity * 40
                    base[..., 1] = 25 + stage_index * 45
                    base[..., 2] = 60 + sample_index * 15
                    noise = rng.integers(0, 20, size=base.shape, dtype=np.uint8)
                    image = Image.fromarray(
                        np.clip(base + noise, 0, 255).astype(np.uint8)
                    )
                    draw = ImageDraw.Draw(image)
                    radius = 5 + identity
                    center = arguments.image_size // 2 + stage_index - 1
                    draw.ellipse(
                        (
                            center - radius,
                            center - radius,
                            center + radius,
                            center + radius,
                        ),
                        outline=(255, 255, 255),
                        width=2,
                    )
                    filename = "{}_id{:03d}_{}_{}.png".format(
                        stage, identity, split, sample_index
                    )
                    path = image_root / filename
                    image.save(path)
                    rows.append(
                        {
                            "path": filename,
                            "identity": "ID{:03d}".format(identity),
                            "stage": stage,
                            "split": split,
                            "group_id": "{}_id{:03d}_{}_{}".format(
                                stage, identity, split, sample_index
                            ),
                            "acquisition_type": "synthetic",
                        }
                    )
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False)
    print(output / "manifest.csv")


if __name__ == "__main__":
    main()
