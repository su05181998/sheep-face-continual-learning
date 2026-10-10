import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public ImageNet-1K-pretrained LSNet-T weights"
    )
    parser.add_argument("--output", default="checkpoints/lsnet_t_imagenet1k.pth")
    parser.add_argument("--repository", default="jameslahm/lsnet_t")
    parser.add_argument("--filename", default="pytorch_model.bin")
    arguments = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit(
            "Install the pretrained extra first: pip install -e '.[pretrained]'"
        ) from error
    downloaded = hf_hub_download(
        repo_id=arguments.repository,
        filename=arguments.filename,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, output)
    print(output)


if __name__ == "__main__":
    main()
