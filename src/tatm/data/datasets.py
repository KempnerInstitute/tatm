"""This module is intended to provide external interfaces to datasets
prepared and/or curated by the TATM project. The datasets are intended
to be consumed by modelling frameworks such as pytorch, JAX, etc.
"""

import json
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


class TatmDataset(ABC):
    """Abstract base class for TATM datasets."""

    @abstractmethod
    def __len__(self):
        """Get the number of tokens in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Get the token at the given index."""
        pass


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
        self._get_file_info()

    def _get_file_info(self):
        """Get the file information."""
        file_size = self.file_path.stat().st_size
        self.num_tokens = file_size // np.dtype(self.dtype).itemsize
        self.array = np.memmap(
            self.file_path, dtype=self.dtype, mode="r", shape=(self.num_tokens,)
        )

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

    def __len__(self):
        """Get the number of examples in the dataset."""
        return self.file_list[-1][0] + len(self.file_list[-1][1])

    def __getitem__(self, idx: int):
        """Get the token at the given index."""
        if idx < 0:
            idx = len(self) + idx
            if idx < 0:
                raise IndexError("Index out of bounds.")

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
        self, img_root: str, ann_paths: list, *, img_processor=None, text_processor=None
    ):
        """
        Args:
            img_root: The root directory containing all of the images used by this dataset
            ann_paths: List of files containing the annotations within this dataset.
                    Each annotation should give the path (relative to img_root) of the image it is describing
            img_processor: Function for preprocessing annotation images within the dataset
            text_processor: Function for preprocessing annotation text within the dataset
        """
        self.img_root = Path(img_root)
        self.img_processor = img_processor
        self.text_processor = text_processor
        self.annotations = []

        for ann_path in ann_paths:
            with open(ann_path, "r") as f:
                self.annotations.extend(json.load(f))

    def __len__(self):
        return len(self.annotations)

    def set_processors(self, *, img_processor=None, text_processor=None):
        self.img_processor = img_processor
        self.text_processor = text_processor


class TatmCaptionedImageDataset(TatmImageTextDataset):
    """
    Handles captioned images. Each annotation should have a text caption and a path to an image that the caption describes.
    """

    def __init__(
        self, img_root: str, ann_paths: list, *, img_processor=None, text_processor=None
    ):
        super().__init__(
            img_root,
            ann_paths,
            img_processor=img_processor,
            text_processor=text_processor,
        )

    def __getitem__(self, index: int):
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
