# Getting Started

## Installation

### Requirements

- Python 3.10 is the minimum supported version. 

- `tatm` depends on pytorch, which depends on CUDA. It is recommended to pre-install both pytorch and CUDA prior to installing `tatm`. Instructions for installing pytorch can be found [here](https://pytorch.org/get-started/locally/).

### Installation

#### Installing from GitHub

To install the latest stable version of `tatm` from GitHub, run the following command:

```bash
pip install git+ssh://git@github.com/KempnerInstitute/tatm.git@main
```

To install the latest development version of `tatm` from GitHub, run the following command:

```bash
pip install git+ssh://git@github.com/KempnerInstitute/tatm.git@dev
```

For a specific past version of `tatm`, replace `main` or `dev` with the desired version number (i.e. `v0.1.0`).

#### Installing from PyPI

The package is not yet available on PyPI. Stay tuned for updates!

## Loading Tokenized Data with `tatm` for use with PyTorch

In the example code below, we show how to create a PyTorch dataloader with a tokenized dataset for use with a model.

```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from tatm.data import get_dataset, torch_collate_fn
tatm_dataset = get_dataset("<PATH TO TATM TOKENIZED DATA>", context_length=1024)
len(tatm_dataset) # number of examples in set
# 35651584
tatm_dataset.num_tokens()
# 36507222016
tatm_dataset.num_files()
# 34
tatm_dataset.vocab_size
# 32100
tatm_dataset[3]
# Note that the output will vary depending on the dataset and the tokenization process as the order documents are tokenized may vary.
# TatmMemmapDatasetItem(
#    token_ids=array([    7,    16,     8, ..., 14780,     8,  2537], dtype=uint16), 
#    document_ids=array([0, 0, 0, ..., 1, 1, 1], dtype=uint16)
# )

dataloader = DataLoader(tatm_dataset, batch_size=4, collate_fn=torch_collate_fn)
print(next(iter(dataloader)))
# {'token_ids': tensor([[    3,     2, 14309,  ...,  1644,  4179,    16],
#         [ 3731,  3229,     2,  ...,    15,     2,     3],
#         [    2, 14309,     2,  ...,   356,     5, 22218],
#         [    7,    16,     8,  ..., 14780,     8,  2537]], dtype=torch.uint16), 
#    'document_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 1, 1, 1]], dtype=torch.uint16)}
```



