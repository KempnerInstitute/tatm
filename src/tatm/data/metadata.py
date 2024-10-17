import dataclasses
import datetime
import json
import os
import pathlib
from enum import Enum
from typing import List, Union

import yaml


@dataclasses.dataclass(kw_only=True)
class TokenizedMetadataComponenet:
    tokenizer: str
    file_prefix: str
    dtype: str = "uint16"
    file_extension: str = "bin"
    vocab_size: int = None
    tatm_version: str = (
        None  #: Version of the tatm library used to create the tokenized data. Default to None to avoid breaking changes/overwriting past versions.
    )
    tokenizers_version: str = (
        None  #: Version of the hf tokenizer used to create the tokenized data. Default to None to avoid breaking changes/overwriting past versions.
    )


class DataContentType(str, Enum):
    """Enum class for dataset content"""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


class CorpusSeparationStrategy(str, Enum):
    """Enum class for corpus separation strategy"""

    DATA_DIRS = "data_dirs"  #: Corpus data is separated into directories. Maps to using the data_dir parameter in the datasets.load_dataset function.
    CONFIGS = "configs"  #: Corpus data is separated into config files. Maps to using the name parameter in the datasets.load_dataset function.

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


@dataclasses.dataclass(kw_only=True)
class TatmDataMetadata:
    """Generic Dataset Metadata Class holding information about a dataset.

    Raises:
        ValueError: Raises a ValueError if the data_content value is invalid.
    """

    name: str  #: Name of the dataset.
    dataset_path: str  #: Path to the dataset.
    description: str  #: Description of the dataset.
    date_downloaded: str  #: Date the dataset was downloaded.
    download_source: str  #: Source of the dataset.
    data_content: DataContentType  #: Type of data in the dataset.
    content_field: str = "text"  #: Field in the dataset that contains the content.
    corpuses: List[str] = dataclasses.field(
        default_factory=list
    )  #: List of corpuses in the dataset.

    # fmt: off
    corpus_separation_strategy: CorpusSeparationStrategy = None  #: Strategy for separating corpuses in the dataset (data_dirs or configs). data_dirs maps to using the data_dir parameter in the datasets.load_dataset function, and configs maps to using the name parameter.
    corpus_data_dir_parent: str = None  #: Parent directory of corpus data directories, to be used with corpus_separation_strategy='data_dirs'. If None, the corpus name is assumed to map to a directory in the top level of the dataset path.
    # fmt: on

    tokenized_info: TokenizedMetadataComponenet = None  #: Metadata for tokenized data.

    def __post_init__(self):
        self._validate()
        self.data_content = DataContentType(self.data_content)
        if self.corpus_separation_strategy is not None:
            self.corpus_separation_strategy = CorpusSeparationStrategy(
                self.corpus_separation_strategy
            )

        if isinstance(self.tokenized_info, dict):
            self.tokenized_info = TokenizedMetadataComponenet(**self.tokenized_info)

    def _validate(self):
        if not DataContentType.has_value(self.data_content):
            raise ValueError(f"Invalid data_content value: {self.data_content}")
        if self.corpus_separation_strategy is not None:
            if not CorpusSeparationStrategy.has_value(self.corpus_separation_strategy):
                raise ValueError(
                    f"Invalid corpus_separation_strategy value: {self.corpus_separation_strategy}"
                )

    def as_json(self):
        """Return the metadata as a JSON string.

        Returns:
            str: Metadata as a JSON string.
        """
        out = {k: v for k, v in dataclasses.asdict(self).items() if v is not None}
        return json.dumps(out)

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
        out = {k: v for k, v in dataclasses.asdict(self).items() if v is not None}
        out["data_content"] = self.data_content.value
        if self.corpus_separation_strategy:
            out["corpus_separation_strategy"] = self.corpus_separation_strategy.value
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

    @classmethod
    def from_file(cls, path: Union[str, pathlib.Path]):
        """Load metadata from a file, either JSON or YAML."""
        if isinstance(path, str):
            path = pathlib.Path(path)
        if path.suffix == ".json":
            return cls.from_json(path)
        elif path.suffix in [".yaml", ".yml"]:
            return cls.from_yaml(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    @classmethod
    def from_directory(cls, directory: Union[str, pathlib.Path]):
        """Load metadata from a directory containing a metadata file named metadata.[yaml or json]."""
        if isinstance(directory, str):
            directory = pathlib.Path(directory)

        json_path = directory / "metadata.json"
        yaml_path = directory / "metadata.yaml"
        yml_path = directory / "metadata.yml"

        if json_path.exists():
            return cls.from_json(json_path)
        elif yaml_path.exists():
            return cls.from_yaml(yaml_path)
        elif yml_path.exists():
            return cls.from_yaml(yml_path)
        else:
            raise ValueError("No metadata file found in the specified directory.")

    def __str__(self):
        return self.as_json()


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

    corpus_separation_strategy = None
    if len(corpuses) > 1:
        while True:
            corpus_separation_strategy = input(
                "Corpus separation strategy ([data_dirs], configs): "
            )
            print(f"corpus_separation_strategy: {corpus_separation_strategy}")
            if not corpus_separation_strategy:

                corpus_separation_strategy = "data_dirs"
                break
            if corpus_separation_strategy not in ["data_dirs", "configs"]:
                print("Invalid input. Please enter 'data_dirs' or 'configs'.")
                continue
            break
        if corpus_separation_strategy == "data_dirs":
            corpus_data_dir_parent = input(
                "Parent directory of corpus data directories: "
            )
            if not corpus_data_dir_parent:
                corpus_data_dir_parent = None

    while True:
        tokenized = input("Is the dataset tokenized? (y/n) [n]:")
        if not tokenized:
            tokenized = "n"
        if tokenized not in ["y", "n"]:
            print("Invalid input. Please enter 'y' or 'n'.")
            continue
        break
    if tokenized == "y":
        tokenized_info = _create_tokenized_metadata_interactive()
    else:
        tokenized_info = None

    metadata = TatmDataMetadata(
        name=name,
        dataset_path=dataset_path,
        description=description,
        date_downloaded=date_downloaded,
        download_source=download_source,
        data_content=DataContentType(data_content),
        content_field=content_field,
        corpuses=corpuses,
        corpus_separation_strategy=corpus_separation_strategy,
        corpus_data_dir_parent=corpus_data_dir_parent,
        tokenized_info=tokenized_info,
    )

    if output_format == "json":
        metadata.to_json(output_path)
    else:
        metadata.to_yaml(output_path)


def _create_tokenized_metadata_interactive() -> TokenizedMetadataComponenet:
    """Contruct a TokenizedDataMetadata object interactively. Intended to be called
    within create_metadata_interactive.

    Returns:
        TokenizedDataMetadata: Tokenized metadata object.
    """

    tokenizer = input("Tokenizer Name or Path to JSON [t5-base]: ")
    if not tokenizer:
        tokenizer = "t5-base"

    file_prefix = input("Token File Prefix [tokenized]: ")
    if not file_prefix:
        file_prefix = "tokenized"

    file_extension = input("Token File Extension [bin]: ")
    if not file_extension:
        file_extension = "bin"

    dtype = input("Token Data Type [uint16]: ")
    if not dtype:
        dtype = "uint16"

    return TokenizedMetadataComponenet(
        tokenizer=tokenizer,
        file_prefix=file_prefix,
        dtype=dtype,
        file_extension=file_extension,
    )
