"""This module is intended to provide external interfaces to datasets
prepared and/or curated by the TATM project. The datasets are intended
to be consumed by modelling frameworks such as pytorch, JAX, etc.
"""

from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path

import numpy as np

from tatm.data.metadata import DataMetadata


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


def get_dataset(metadata: DataMetadata) -> TatmDataset:
    """Get the dataset object from the metadata.

    Args:
        metadata: The metadata object.

    Returns:
        TatmDataset: The dataset object.
    """
    if metadata.tokenized_info:
        return TatmMemmapDataset(
            metadata.tokenized_info["file_prefix"],
            metadata.tokenized_info["context_length"],
            metadata.tokenized_info["dtype"],
            metadata.tokenized_info["chunked"],
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
        """Get the number of tokens in the array."""
        if self.chunked:
            return self.num_tokens // self.context_length
        else:
            return self.num_tokens

    def __getitem__(self, idx):
        """Get the token at the given index."""
        if self.chunked:
            return self.array[
                idx * self.context_length : (idx + 1) * self.context_length
            ]
        else:
            return self.array[idx : idx + self.context_length]


class TatmMemmapDataset(TatmDataset):
    """Class for presenting tatm tokenized datasets to modelling frameworks."""

    def __init__(
        self,
        file_prefix: str,
        context_length: int,
        dtype: str = "uint16",
        chunked: bool = True,
        file_suffix: str = "bin",
    ):
        """Initialize the TatmTokenizedDataset.

        Args:
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
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.context_length = context_length
        self.dtype = dtype
        self.chunked = chunked
        self._construct_file_list()

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
        """Get the number of tokens in the dataset."""
        return self.file_list[-1][0] + len(self.file_list[-1][1])

    def __getitem__(self, idx: int):
        """Get the token at the given index."""
        for start, array in self.file_list:
            if idx < start + len(array):
                # Linear search right now, can be optimized with binary search, but feels premature atm.
                return self._process_item(array[idx - start])
        raise IndexError("Index out of bounds.")

    def _process_item(self, item):
        """Process the item. Currently a no-op."""
        return item

    def num_files(self):
        """Get the number of files in the dataset."""
        return len(self.file_list)

    def num_tokens(self):
        """Get the number of tokens in the dataset."""
        return sum([i.num_tokens for _, i in self.file_list])
