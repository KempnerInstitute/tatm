import dataclasses
import json
import pathlib
import yaml


@dataclasses.dataclass
class Metadata:
    name: str
    dataset_path: str
    description: str
    date_downloaded: str
    download_source: str
    data_content: str
    content_field: str = "text"

    def as_json(self):
        """Return the metadata as a JSON string.

        Returns:
            str: Metadata as a JSON string.
        """
        return json.dumps(dataclasses.asdict(self))

    def to_json(self, filename):
        """Write the metadata to a JSON file.

        Args:
            filename (str): The path of the file to write the metadata to.
        """
        with open(filename, "w") as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def from_json(cls, json_path):
        """Create a Metadata object from a JSON file."""
        with open(json_path, "r") as f:
            metadata = json.load(f)

        if "dataset_path" not in metadata:
            parent_dir = pathlib.Path(json_path).resolve().parent
            metadata["dataset_path"] = str(parent_dir)

        return cls(**metadata)

    def as_yaml(self):
        return yaml.dump(dataclasses.asdict(self))

    def to_yaml(self, filename):
        with open(filename, "w") as f:
            yaml.dump(dataclasses.asdict(self), f)

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            metadata = yaml.safe_load(f)

        if "dataset_path" not in metadata:
            parent_dir = pathlib.Path(yaml_path).resolve().parent
            metadata["dataset_path"] = str(parent_dir)
        return cls(**metadata)
