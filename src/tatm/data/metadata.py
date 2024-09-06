import dataclasses
import datetime
import json
import os
import pathlib
from enum import Enum
from typing import List

import yaml


class DatasetContentType(str, Enum):
    """Enum class for dataset content"""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


@dataclasses.dataclass
class DatasetMetadata:
    """Generic Dataset Metadata Class holding information about a dataset.

    Raises:
        ValueError: Raises a ValueError if the data_content value is invalid.
    """

    name: str  #: Name of the dataset.
    dataset_path: str  #: Path to the dataset.
    description: str  #: Description of the dataset.
    date_downloaded: str  #: Date the dataset was downloaded.
    download_source: str  #: Source of the dataset.
    data_content: DatasetContentType  #: Type of data in the dataset.
    content_field: str = "text"  #: Field in the dataset that contains the content.
    corpuses: List[str] = dataclasses.field(
        default_factory=list
    )  #: List of corpuses in the dataset.

    def __post_init__(self):
        self._validate()
        self.data_content = DatasetContentType(self.data_content)

    def _validate(self):
        if not DatasetContentType.has_value(self.data_content):
            raise ValueError(f"Invalid data_content value: {self.data_content}")

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
            f.write(self.as_json())

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
        out = dataclasses.asdict(self)
        out["data_content"] = self.data_content.value
        return yaml.dump(out)

    def to_yaml(self, filename):
        with open(filename, "w") as f:
            f.write(self.as_yaml())

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            metadata = yaml.safe_load(f)

        if "dataset_path" not in metadata:
            parent_dir = pathlib.Path(yaml_path).resolve().parent
            metadata["dataset_path"] = str(parent_dir)
        return cls(**metadata)


def create_metadata_interactive():
    """Construct a Metadata object interactively."""

    output_format = input("Output format ([json], yaml): ")
    if not output_format:
        output_format = "json"
    if output_format not in ["json", "yaml"]:
        raise ValueError("Invalid output format value.")

    current_dir = os.getcwd()

    output_path = input(f"Output path [{current_dir}/metadata.{output_format}]: ")
    if not output_path:
        output_path = f"{current_dir}/metadata.{output_format}"

    name = input("Name: ")

    dataset_path = input(f"Dataset path [{current_dir}]: ")
    if not dataset_path:
        dataset_path = current_dir

    description = input("Dataset Description: ")

    cur_date = datetime.datetime.now().strftime("%Y-%m-%d")
    date_downloaded = input(f"Date downloaded [{cur_date}]: ")
    if not date_downloaded:
        date_downloaded = cur_date

    download_source = input("Download source: ")

    data_content = input("Data content ([text], image, audio, video, other): ")
    if not data_content:
        data_content = "text"
    if data_content not in ["text", "image", "audio", "video", "other"]:
        raise ValueError("Invalid data content value.")

    content_field = input("Content field [text]: ")
    if not content_field:
        content_field = "text"

    corpuses = []
    while True:
        if corpuses:
            print("Current corpuses: ", corpuses)
            corpus = input("Add another corpus (leave blank to finish): ")
        else:
            corpus = input("List any corpuses in the dataset (leave blank to finish): ")
        if not corpus:
            break
        corpuses.append(corpus)

    metadata = DatasetMetadata(
        name=name,
        dataset_path=dataset_path,
        description=description,
        date_downloaded=date_downloaded,
        download_source=download_source,
        data_content=DatasetContentType(data_content),
        content_field=content_field,
        corpuses=corpuses,
    )

    if output_format == "json":
        metadata.to_json(output_path)
    else:
        metadata.to_yaml(output_path)
