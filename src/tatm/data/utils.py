import dataclasses

import numpy as np
import torch

from tatm.data.datasets import TatmDatasetItem


def torch_collate_fn(batch: list[TatmDatasetItem]) -> dict[str, torch.Tensor]:
    """Collate function for torch DataLoader. Assumes that all items in the batch are of the same type."""
    out = {}
    for key in dataclasses.asdict(batch[0]).keys():
        if batch[0][key] is not None:
            if isinstance(batch[0][key], np.ndarray):
                out[key] = torch.stack([torch.from_numpy(item[key]) for item in batch])
            elif isinstance(batch[0][key], torch.Tensor):
                out[key] = torch.stack([item[key] for item in batch])
            else:
                out[key] = [item[key] for item in batch]
    return out
