"""This module is intended to provide external interfaces to datasets
prepared and/or curated by the TATM project. The datasets are intended
to be consumed by modelling frameworks such as pytorch, JAX, etc.
"""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from tatm.data.metadata import TatmDataMetadata
from tatm.utils import TatmOptionEnum


class SplitType(TatmOptionEnum):
    """Enum for split types."""

    TRAIN = "train"
    VALIDATION = "validation"


class TatmDataset(ABC):
    """Abstract base class for TATM datasets."""

    def __init__(
        self,
        *,
        split: Optional[SplitType] = None,
        val_split_size: Optional[Union[float, int]] = None,
    ):
        """Initialize the TatmDataset.

        Args:
            split (optional): The split of the data that the __len__ and __getitem__ will operate on. If None, __len__ and __getitem__ will use the
                unsplit dataset. This can be adjusted at runtime by using the set_split method. If defined here,
                val_split_size must also be defined. Defaults to None.
            val_split_size (optional): The size of the validation split. If less than 1, assumed to be a ratio, if
                greater than one assumed to be an observation count. Defaults to None.
        """
        self.split = None  # Initialize the split to None to prevent errors when calling create_split before set_split
        if split is not None and val_split_size is None:
            raise ValueError(
                "If split is defined, val_split_size must also be defined."
            )
        if val_split_size is not None:
            self.create_split(val_split_size)

        self.set_split(split)

    @abstractmethod
    def _num_samples(self) -> int:
        """Get the number of examples in the dataset, regardless of split."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def __len__(self) -> int:
        """Get the number of examples in the current split of the dataset."""
        if self.split is None:
            return self._num_samples()
        elif self.split == SplitType.TRAIN:
            return self._split_index
        elif self.split == SplitType.VALIDATION:
            return self._num_samples() - self._split_index

    @abstractmethod
    def _get(self, idx: int):
        """Retrieve the item at the given global index within the dataset, regardless of split."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def __getitem__(self, idx: int):
        """Get the token at the given index. If a split is defined, the index within the split will be converted to a global index."""
        if idx < 0:
            idx = len(self) + idx
            if idx < 0:
                raise IndexError("Index out of bounds.")

        if idx >= len(self):
            raise IndexError("Index out of bounds.")

        if self.split == SplitType.VALIDATION:
            idx += self._split_index

        return self._get(idx)

    def create_split(self, split_size: Union[float, int] = 0.1):
        """Determine an index to split the dataset into training and validation sets.
        Splits the dataset into a training and validation set based on the split size where the last
        indices are used for validation.
        Sets the index that is the first index of the validation set.

        Args:
            split_size: Either the ratio of the validation set to the whole data or
                the number of observations in the validation set. If less than 1,
                assumed to be a ratio, if greater than one assumed to be an
                observation account. Defaults to 0.1.
        """
        current_split = self.split
        self.set_split(
            None
        )  # Reset the split so that the whole dataset length is used to determine the split
        if split_size < 1:
            split_size = math.ceil(len(self) * split_size)
        self._split_index = len(self) - split_size
        self.set_split(current_split)

    def set_split(self, split: Optional[SplitType] = None):
        """Set the split of the data that the __len__ and __getitem__ will operate on. If called without an argument, __len__ and __getitem__ will use the
        unsplit dataset.

        Args:
            split: The split of the data that the __len__ and __getitem__ will operate on. If None, __len__ and __getitem__ will use the
                unsplit dataset. Defaults to None.
        """
        if split is not None:
            if not SplitType.has_value(split):
                raise ValueError(
                    f"Invalid split type {split}. Valid values are {SplitType.values()}."
                )
            if self._split_index is None:
                raise ValueError(
                    "No current index to split the dataset has been set. Please call create_split prior to setting a split."
                )
            self.split = SplitType(split)
        else:
            self.split = None


def get_dataset(metadata: Union[str, TatmDataMetadata], **kwargs) -> TatmDataset:
    """Get the dataset object from the metadata.

    Args:
        metadata: The metadata object, or a path to a metadata file or a directory containing a metadata file.
        **kwargs: Additional arguments to pass to the dataset constructor.


    Returns:
        TatmDataset: The dataset object.
    """
    metadata_loaded = False
    if isinstance(metadata, str):
        if metadata[0] != "/" and metadata[0] != ".":
            try:
                metadata = TatmDataMetadata.from_metadata_store(metadata)
                metadata_loaded = True
            except ValueError:
                pass
        if not metadata_loaded:
            metadata = Path(metadata)
            if not metadata.exists():
                raise FileNotFoundError(
                    f"Metadata file or directory not found at {metadata}"
                )
            if metadata.is_dir():
                metadata = TatmDataMetadata.from_directory(metadata)
            elif metadata.is_file():
                metadata = TatmDataMetadata.from_file(metadata)
    if metadata.tokenized_info:
        return TatmMemmapDataset(
            file_prefix=metadata.tokenized_info.file_prefix,
            dtype=metadata.tokenized_info.dtype,
            vocab_size=metadata.tokenized_info.vocab_size,
            **kwargs,
        )
    raise NotImplementedError(
        "Metadata does not describe a model ready tatm dataset type."
    )


class TokenMemMapArray:
    """Class for interacting with individual memory mapped tokenized arrays."""

    def __init__(
        self,
        file_path: str,
        context_length: int,
        dtype: str = "uint16",
        chunked: bool = True,
    ):
        """Initialize the TokenMemMapArray.

        Args:
            file_path: Path to the memory mapped array file.
            context_length: The context length of the model.
            dtype: The data type of the array.
            chunked: Whether or not indices should map to chunks of context length of individual tokens.
                     This determines whether or not tokens are seen at least once by the model in the entirity
                     of their context. If True, the underlying array is broken into logical sections of
                    context_length tokens. If False, every token can be the beginning of a context. There is a
                    tradeoff between time to process an epoch and the ability of the model to see tokens in their
                    full context. In our thinking, we have determined that for foundational models, the ability
                    to see tokens in their full context is less important than the time to process an epoch and
                    have chosen to use the chunked approach by default.
        """
        self.file_path = Path(file_path)
        self.context_length = context_length
        self.dtype = dtype
        self.chunked = chunked
        self.array = None
        self._get_file_info()

    def _get_file_info(self):
        """Get the file information."""
        file_size = self.file_path.stat().st_size
        self.num_tokens = file_size // np.dtype(self.dtype).itemsize

    def __len__(self):
        """Get the number of tokens or chunks in the array."""
        if self.chunked:
            chunks = self.num_tokens // self.context_length
            if self.num_tokens % self.context_length != 0:
                chunks += 1
            return chunks
        else:
            return self.num_tokens

    def __getitem__(self, idx):
        """
        Get the token or chunk at the given index.

        The last chunk will be right-padded if there are
        not enough tokens left to fill it.
        """
        if self.array is None:
            self._open_array()

        if self.chunked:
            chunk = self.array[
                idx * self.context_length : (idx + 1) * self.context_length
            ]
            return np.pad(
                chunk,
                (0, (self.context_length - len(chunk)) % self.context_length),
                "constant",
            )
        else:
            return self.array[idx : idx + self.context_length]

    def __getstate__(self):
        """Get the state for pickling.
        We don't want to pickle the array, so we close it before pickling."""
        self.close_array()
        return super().__getstate__()

    def _open_array(self):
        """Open the memory map."""
        self.array = np.memmap(
            self.file_path, dtype=self.dtype, mode="r", shape=(self.num_tokens,)
        )

    def close_array(self):
        """Close the memory map."""
        del self.array
        self.array = None


@dataclass
class TatmDatasetItem:
    """Base class for representing a single item in a TatmDataset."""

    def __getitem__(self, item):
        """Dict like access to attributes."""
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(f"{item} not found in dataset item.")


@dataclass(kw_only=True)
class TatmMemmapDatasetItem(TatmDatasetItem):
    """Class for representing a single item in the TatmMemmapDataset.
    Includes __getitem__ method for dictlike access to the tokenized data."""

    token_ids: Union[np.ndarray, torch.Tensor] = None
    document_ids: Optional[np.ndarray] = None
    document_mask: Optional[np.ndarray] = None


class TokenOutputFormat(TatmOptionEnum):
    """Enum for token output format."""

    TORCH = "torch"
    NP = "numpy"


class TatmMemmapDataset(TatmDataset):
    """Class for presenting tatm tokenized datasets to modelling frameworks."""

    def __init__(
        self,
        file_prefix: str,
        context_length: int,
        dtype: str = "uint16",
        chunked: bool = True,
        file_suffix: str = "bin",
        eos_token: int = 1,
        token_output_format: TokenOutputFormat = TokenOutputFormat.TORCH,
        vocab_size: Union[int, None] = None,
        create_doc_ids: bool = True,
        create_doc_mask: bool = False,
        **kwargs,
    ):
        """Initialize the TatmTokenizedDataset.

        Args:
            file_prefix: Prefix for the tokenized files, shoud be the absolute path, including the parent directory.
            context_length: The context length of the model.
            dtype: The data type of the array.
            chunked: Whether or not indices should map to chunks of context length of individual tokens.
                     This determines whether or not tokens are seen at least once by the model in the entirity
                     of their context. If True, the underlying array is broken into logical sections of
                    context_length tokens. If False, every token can be the beginning of a context. There is a
                    tradeoff between time to process an epoch and the ability of the model to see tokens in their
                    full context. In our thinking, we have determined that for foundational models, the ability
                    to see tokens in their full context is less important than the time to process an epoch and
                    have chosen to use the chunked approach by default.
            file_suffix: Suffix for the tokenized files.
            eos_token: The end of sequence token ID.
            vocab_size (optional): The vocabulary size of the tokenizer used to create the dataset.
            create_doc_ids (optional): Whether or not to create document ids (IDs linking tokens to each local documents, based on the EOS token). Defaults to True.
            create_doc_mask (optional): Whether or not to create a document mask (mask for attention based on document IDs). Defaults to False. Note that this incurs a memory overhead and significant
                performance hit in the current implementation. Requires create_doc_ids to be True.
            kwargs: Additional arguments to pass to the dataset constructor.
        """
        self.token_output_format = token_output_format
        self._validate()
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.context_length = context_length
        self.dtype = dtype
        self.chunked = chunked
        self.eos_token = eos_token
        self.vocab_size = vocab_size
        self.create_doc_ids = create_doc_ids
        self.create_doc_mask = create_doc_mask
        if self.create_doc_mask and not self.create_doc_ids:
            raise ValueError(
                "Document mask creation requires create_doc_ids to be True."
            )
        self._construct_file_list()

        super().__init__(**kwargs)

    def _validate(self):
        """Validate the passed in inputs"""
        if not TokenOutputFormat.has_value(self.token_output_format):
            raise ValueError(
                f"Invalid token output format {self.token_output_format}. Valid values are {TokenOutputFormat.values()}."
            )
        if not isinstance(self.token_output_format, TokenOutputFormat):
            self.token_output_format = TokenOutputFormat(self.token_output_format)

    def _construct_file_list(self):
        """Construct the list of tokenized files."""
        file_list = glob(f"{self.file_prefix}*.{self.file_suffix}")
        file_list.sort()
        file_list = [
            TokenMemMapArray(i, self.context_length, self.dtype, self.chunked)
            for i in file_list
        ]

        self.file_list = []
        start_idx = 0
        for i in file_list:
            self.file_list.append((start_idx, i))
            start_idx += len(i)

    def _num_samples(self):
        """Get the number of examples in the dataset, regardless of split."""
        return self.file_list[-1][0] + len(self.file_list[-1][1])

    def _get(self, idx: int):
        """Get the token at the given index."""

        for start, array in self.file_list:
            if idx < start + len(array):
                # Linear search right now, can be optimized with binary search, but feels premature atm.
                return self._process_item(array[idx - start])
        raise IndexError("Index out of bounds.")

    def _process_item(self, item):
        """Process the item. Construct item response."""

        out = TatmMemmapDatasetItem(
            token_ids=self._format_output_tokens(item),
        )
        if self.create_doc_ids:
            doc_ids = self._format_output_tokens(
                _get_document_ids(item, eos_token=self.eos_token)
            )
            out.document_ids = doc_ids
        if self.create_doc_mask:
            doc_mask = self._format_output_tokens(_create_document_mask(doc_ids))
            out.document_mask = doc_mask
        return out

    def _format_output_tokens(self, item):
        """Format the output tokens."""
        if self.token_output_format == TokenOutputFormat.TORCH:
            return torch.tensor(item, dtype=torch.long)
        elif self.token_output_format == TokenOutputFormat.NP:
            return item
        else:
            raise NotImplementedError(
                f"Token output format {self.token_output_format} not supported."
            )

    def num_files(self):
        """Get the number of files in the dataset."""
        return len(self.file_list)

    def num_tokens(self):
        """Get the number of tokens in the dataset."""
        return sum([i.num_tokens for _, i in self.file_list])


def _get_document_ids(tokens: np.ndarray, eos_token: int = 1) -> np.ndarray:
    """Get document ids from token ids."""
    document_ids = np.zeros_like(tokens, dtype=np.uint16)
    current_doc_id = 0
    for i in range(len(tokens)):
        document_ids[i] = current_doc_id
        if tokens[i] == eos_token:
            current_doc_id += 1

    return document_ids


def _create_document_mask(doc_ids: np.ndarray) -> np.ndarray:
    """Create a document based attention mask from document ids."""
    document_equal = np.equal.outer(doc_ids, doc_ids)
    out = np.logical_and(document_equal, np.tri(len(doc_ids)))
    return out


class TatmImageTextDataset(TatmDataset):
    """
    Base class for handling all image-text datasets. This includes captioned image datasets and image question-answer (often denoted VQA) datasets.
    """

    def __init__(
        self,
        img_root: str,
        ann_paths: list[Union[Path, str]],
        *,
        img_processor=None,
        text_processor=None,
        **kwargs,
    ):
        """
        Args:
            img_root: The root directory containing all of the images used by this dataset
            ann_paths: List of files containing the annotations within this dataset.
                    Each annotation should give the path (relative to img_root) of the image it is describing
            img_processor: Function for preprocessing annotation images within the dataset
            text_processor: Function for preprocessing annotation text within the dataset
            **kwargs: Additional arguments to pass to the dataset constructor
        """
        self.img_root = Path(img_root)
        self.img_processor = img_processor
        self.text_processor = text_processor
        self.annotations = []

        self._load_annotations(ann_paths)

        super().__init__(**kwargs)

    def _load_annotations(self, ann_paths: list[Union[Path, str]]):
        """
        Handles loading annotations into the dataset. Currently handles json and jsonl files.

        Args:
            ann_paths: List of files containing the annotations within this dataset.
                    Each annotation should give the path (relative to img_root) of the image it is describing
        """
        for ann_path in ann_paths:
            ann_path = Path(ann_path)
            if ann_path.suffix == ".json":
                with ann_path.open(mode="r") as f:
                    self.annotations.extend(json.load(f))
            elif ann_path.suffix == ".jsonl":
                with ann_path.open(mode="r") as f:
                    self.annotations.extend([json.loads(line) for line in f])
            else:
                raise ValueError(
                    "Only .json and .jsonl files are supported for image-text annotations."
                )

    def _num_samples(self):
        return len(self.annotations)

    def set_processors(self, *, img_processor=None, text_processor=None):
        self.img_processor = img_processor
        self.text_processor = text_processor


class TatmCaptionedImageDataset(TatmImageTextDataset):
    """
    Handles captioned images. Each annotation should have a text caption and a path to an image that the caption describes.
    """

    def __init__(
        self,
        img_root: str,
        ann_paths: list[Union[Path, str]],
        *,
        img_processor=None,
        text_processor=None,
        **kwargs,
    ):
        super().__init__(
            img_root,
            ann_paths,
            img_processor=img_processor,
            text_processor=text_processor,
            **kwargs,
        )

    def _get(self, index: int):
        """
        Retrieves the caption and image of the requested annotation.

        Args:
            index: Index of the annotation to retrieve
        """
        ann = self.annotations[index]

        image_path = self.img_root / ann["image"]
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None  # image does not exist

        image = self.img_processor(image)
        caption = self.text_processor(ann["caption"])

        return {"image": image, "caption": caption, "image_id": ann["image_id"]}
