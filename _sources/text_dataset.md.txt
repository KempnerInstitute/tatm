# Loading Text Data for LLM Training

## Preparing Raw Data for use with `tatm`

The `tatm` library provides tools for LLM and AI/ML training frameworks to access and use data stored on disk. In order for `tatm` to work with your data, you will need to
create a metadata file that describes the data and how it is stored. 

The `tatm` library provides an interactive CLI tool that can help you create this metadata file. To use this tool,
run the following command from the directory where your data is stored:

```bash
tatm data create-metadata
```

The CLI tool will prompt you for information about your data, such as the name of the dataset, the path to the
raw data files, and the format of the data. The tool will then create a metadata file that describes the data
and how it is stored on disk. The `tatm` library uses this metadata file to load and process the data.

`tatm` assumes that your data is stored in a format that the huggingface `datasets` library can load. 
If your data is not in a format supported by `datasets`, you should create a custom dataset script that can load your data. 
More details on how to structure data for use with `datasets` can be found [here](https://huggingface.co/docs/datasets/en/loading).

## Tokenizing Raw Text Data with `tatm`

To train a LLM on text data, you must first convert the raw text data into
tokenized data that the LLM can interpret. The `tatm` library includes functionality for creating 
arrays of tokens on disk that can be fed directly into a LLM training framework.

`tatm` runs using a `ray` based backend that can parallelize the tokenization across multiple CPUs and multiple nodes,
enabling large datasets to be tokenized quickly and efficiently. `tatm` also includes functionality (`tatm run`) for 
interfacing with SLURM to submit tokenization jobs to a cluster with the proper settings and configuration.

### Setting up your `tatm` Configuration File

To interface with SLURM and define your compute environment, `tatm` utilizes a configuration file that defines the SLURM partition, account, and other tokenization process settings.  The file is passed to the `tatm run` command.

Below is an example configuration file with tatm installed in a conda environment named `tatm_conda`:

```yaml
# Filename: $PWD/tatm_config.yaml
environment:
    modules:
        - python/3.10.13-fasrc01 # maps to python/3.10.13-fasrc01
    conda_env: tatm_conda # Name of the conda environment to use, also works with full paths to the conda environment
slurm:
    partition: example # SLURM partition to use for the job
    account: example # SLURM account to use for the job
```

For full details on how to configure `tatm`, see the [Configuring `tatm`](config.md) documentation.

### Running the tokenization process

To run the tokenizer on SLURM, use the the command `tatm run` with the `tokenize` subcommand and the 
appropriate arguments/options. `tatm` will create a submission script based on the configuration file and run time options,
wrap the `ray` based tokenization process in a SLURM job, and submit the job to the cluster. The options available
to the `tatm run` command are documented in the [CLI](cli.md) documentation and mirror the flags available to the `sbatch` command.

To review the submission script before submitting the job, use the `--no-submit` flag to prevent the job from being submitted.
The submission script will be created in the current working directory and will be named `tatm_tokenize.submit`. The executed `sbatch` command will output to the console.

The tokenization script uses a Ray backend to spin up multiple CPU based tokenization workers and process examples into sequences of tokens in parallel. By default, the number of
workers is determined automatically by the resources available to the Ray cluster. You can specify the a different number of workers to use with the `--num-workers` flag. 

The command below shows an example `tatm run` command to tokenize a dataset. It creates a 4 node ray cluster with 40 CPUs per node
to tokenize the dataset located at `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1:arxiv` and outputs the tokenized data to a directory named `tokenized_redpj_arxiv1` 
in the current working directory. Note that the data at `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1` has already been prepared with a metadata file. The colon `:` specifies the `arxiv` corpus within the dataset. The handling of sub-corpora, implemented by the huggingface dataset script, is dataset-specific and may not be supported by all datasets.

```bash
tatm run --conf $PWD/tatm_config.yaml -N 4 -c 40 tokenize \
  --output-dir $PWD/tokenized_redpj_arxiv \
  /n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1:arxiv
```

This will submit a slurm job creating the Ray cluster.  The `tokenize` command will utilize the Ray cluster to tokenize `arxiv` corpus the dataset located at `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1` and output the tokenized data to the directory `tokenized_redpj_arxiv` in the current working directory.  This will also create a metadata file associated with the tokenized data that can be used to load the tokenized data into a PyTorch model for training. The metadata file, `metadata.json`,will be located in the output directory. It will also include
information about the tokenizer, including the tokenizer's vocabulary and configuration, as well as the version of Huggingface `tokenizers` and `tatm` used to tokenize the data.

By default the `tokenize` command uses the `t5-base` tokenizer from Huggingface. You can specify a different tokenizer to use with the `--tokenizer` flag. You can either pass the name of a tokenizer available from HuggingFace or pass the path to a huggingface compatible tokenizer json file.

### Finding and Selecting Data available within the Testbed

The Kempner AI Testbed provides access to a variety of datasets that can be used for training and evaluation of LLMs. We are in the ongoing process of curating and preparing these datasets for use with the `tatm` library.
In a future release, we will make available a metadata service that will enable users to search for and access datasets available within the testbed, as well as allowing users to easily get information on what corpuses and
tokenized versions are available for a given dataset. For now, for specific dataset questions please reach out to the Kempner Research and Engineering team.

For now, a list of available corpora for a dataset can be found in the metadata for prepared datasets. Note that a corpus tends to be a 1:1 mapping to the "name" concept within a Huggingface dataset. The handling of corpora is implemented by the huggingface dataset script, is dataset-specific and may not be supported by all datasets.

For specifying a corpus of a given dataset the current syntax is `<DATASET_PATH>[:<CORPUS_NAME>]`. The `:` is used to specify the corpus name. If no corpus is specified, the default corpus will be used.

## Loading Tokenized Data with `tatm` for use with PyTorch

Once you have tokenized your data, you can load it into a PyTorch dataset using the `tatm` library. The `tatm` library
provides a PyTorch compatible dataset class that can be used to load tokenized data into a PyTorch model for training
([`tatm.data.TatmMemmapDataset`](tatm.data.TatmMemmapDataset)). You can then load the dataset into a PyTorch `DataLoader` and use it to train your
model. The `TatmMemmapDataset` implements the appropriate `__getitem__` and `__len__` methods to be compatible with PyTorch's
`Dataset` API and supports integration with the Pytorch DistrubutedSampler for distributed training.

In the example code below, we show how to create a PyTorch dataloader with a tokenized dataset for use with a model.

```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from tatm.data import get_dataset, torch_collate_fn
arxiv_dataset = get_dataset("./tokenized_redpj_arxiv", context_length=1024)
len(arxiv_dataset) # number of examples in set
# 35651584
arxiv_dataset.num_tokens()
# 36507222016
arxiv_dataset.num_files()
# 34
arxiv_dataset.vocab_size
# 32100
arxiv_dataset[3]
# Note that the output will vary depending on the dataset and the tokenization process as the order documents are tokenized may vary.
# TatmMemmapDatasetItem(
#    token_ids=array([    7,    16,     8, ..., 14780,     8,  2537], dtype=uint16), 
#    document_ids=array([0, 0, 0, ..., 1, 1, 1], dtype=uint16)
# )

dataloader = DataLoader(arxiv_dataset, batch_size=4, collate_fn=torch_collate_fn)
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

Fields in the `TatmMemmapDatasetItem` object include:

- `token_ids`: The tokenized text data

- `document_ids` (Optional): The document ids for each token. We use example packing to ease the processing of the data in the LLM. To support document masking, we include the document ids for each token in the dataset. Included by default to support document masking.

- `document_mask` (Optional): A boolean attention mask that can be used for causal data masking. This masks tokens that are not part of the same document as the current token, as well as tokens that should not be considered in a given token's attention calculation. Excluded by default for performance reasons.

For more information on how to use the [`tatm.data.TatmMemmapDataset`](tatm.data.TatmMemmapDataset) class, see the [Data](tatm.data.TatmMemmapDataset) documentation.

The provided [`torch_collate_fn`](tatm.data.torch_collate_fn) function is used to collate the data into a batch for training. The function will create stacked tensors or lists for the fields in
the returned `TatmMemmapDatasetItem` object and return a dictionary with the same key names as the dataset item pointing to the stacked items.
