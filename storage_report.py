import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate historical-data storage payloads"
    )
    parser.add_argument("--replay-images", type=int, default=1800)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--identities", type=int, default=180)
    parser.add_argument("--learned-stages", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    arguments = parser.parse_args()
    replay_bytes = arguments.replay_images * arguments.image_size**2 * 3
    prototype_bytes = (
        arguments.identities * arguments.learned_stages * arguments.embedding_dim * 4
    )
    report = {
        "replay": {
            "bytes": replay_bytes,
            "mib": replay_bytes / 1024**2,
        },
        "prototype_memory": {
            "bytes": prototype_bytes,
            "mib": prototype_bytes / 1024**2,
        },
        "payload_ratio_replay_to_prototypes": replay_bytes / prototype_bytes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
