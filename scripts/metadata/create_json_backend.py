import argparse
import json

from utils import metadata_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create JSON metadata backend")
    parser.add_argument(
        "--dirs",
        type=str,
        help="Comma seperated list of directories containing metadata files",
    )
    parser.add_argument("--output", type=str, help="Output JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_dirs = args.dirs.split(",")
    metadata_store = {}
    for dir in metadata_dirs:
        for _, path, metadata in metadata_files(dir):
            if metadata.name in metadata_store:
                raise ValueError(
                    f"Duplicate metadata name found in {path}: {metadata.name}"
                )
            metadata_store[metadata.name] = metadata.as_json()

    with open(args.output, "w") as f:
        json.dump(metadata_store, f)


if __name__ == "__main__":
    main()
