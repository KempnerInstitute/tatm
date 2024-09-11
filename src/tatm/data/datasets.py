"""This module is intended to provide external interfaces to datasets
prepared and/or curated by the TATM project. The datasets are intended
to be consumed by modelling frameworks such as pytorch, JAX, etc.
"""

from abc import ABC, abstractmethod


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
