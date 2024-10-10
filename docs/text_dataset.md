# Loading Text Data for LLM Training

## Preparing Raw Data for use with `tatm`

The `tatm` library provides a set of tools for interacting with data on disk in a standard way that can present
data to various LLM tools and trainig framworks. In order for `tatm` to work with your data, you will need to
create a metadata file that describes the data and how it is stored on disk. 

The `tatm` library provides an interactive CLI tool that can help you create this metadata file. To use this tool,
run the following command from the directory where your data is stored:

```bash
tatm data create-metadata
```

The CLI tool will prompt you for information about your data, such as the name of the dataset, the path to the
raw data files, and the format of the data. The tool will then create a metadata file that describes the data
and how it is stored on disk. This metadata file can be used by the `tatm` library to load and process the data.

`tatm` also assumes that your data is stored in a format that can be loaded by using the huggingface `datasets`
library. This means that your data should either be one of the standard formats supported by `datasets`, or you
should create a custom dataset script that can load your data. More details on how to structure data for use
with `datasets` can be found [here](https://huggingface.co/docs/datasets/en/loading).

## Tokenizing Raw Text Data with `tatm`

In order to train a large language model (LLM) on text data, you will need to convert the raw text data into
tokenized data that can be interpreted by the LLM. The `tatm` library includes functionality for creating 
arrays of tokens on disk that can be fed directly into a LLM training framework for training.

In order to do this efficiently, `tatm` runs using a `ray` based backend that can parallelize the tokenization
across multiple CPUs and multiple nodes enabling you to tokenize large datasets quickly and efficiently. `tatm` also
includes functionality (`tatm run`) for interfacing with SLURM to submit tokenization jobs to a cluster with the proper settings
and configuration.

### Setting up your `tatm` Configuration File

In order to know how to interface with SLURM and how to define your compute environment, `tatm` utilizes a configuration
file. In order to use the `tatm run`, you will need to create a configuration file that defines the SLURM partition, account,
and other settings that are necessary for running the tokenization process and then pass that configuration file to the `tatm run` command.

An example configuration file with tatm installed in a conda environment named `tatm_conda` might look like this:

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

In order to run the tokenizer on slurm, you'll use the the command `tatm run` with the `tokenize` subcommand and the 
appropriate arguments/options. Tatm will create a submission script based on the configuration file and the run time options
wrapping the `ray` based tokenization process in a SLURM job and then submit the job to the cluster. The options available
to the `tatm run` command are documented in the [CLI](cli.md) documentation and mirror the flags available to the `sbatch` command.

If you want to review the submission script before submitting the job, you can use the `--no-submit` flag to prevent the job from being submitted.
The submission script will be created in the current working directory and will be named `tatm_tokenize.submit`. The executed `sbatch` command will be output to the console.

The tokenization script uses a Ray backend to spin up multiple CPU based tokenization workers and process examples into sequences of tokens in parallel. By default, the number of
workers is determined automatically by the resources available to the Ray cluster. You can specify the a different number of workers to use with the `--num-workers` flag. 

The command below shows an example of what using the `tatm run` command to tokenize a dataset might look like. It creates a 4 node ray cluster with 40 CPUs per node
to tokenize the dataset located at `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1:arxiv` and outputs the tokenized data to a directory named `tokenized_redpj_arxiv1` 
in the current working directory. Note that the data at `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1 has already been prepared with a metadata file. We are also
using the colon `:` to specify the `arxiv` corpus within the dataset. The handling of sub-corpora is dataset specific and may not be supported by all datasets and is implemented by the
custom huggingface dataset script.
```bash
tatm run --conf $PWD/tatm_config.yaml -N 4 -c 40 tokenize --output-dir $PWD/tokenized_redpj_arxiv /n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1:arxiv
```

This will submit a slurm job creating the Ray cluster and then execute the `tokenize` command which will utilize the Ray cluster to tokenize the dataset located at `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/redpajama-v1:arxiv` and output the tokenized data to the directory `tokenized_redpj_arxiv` in the current working directory. Note that this will also create a metadata file associated with the tokenized data that can be used to load the tokenized data into a PyTorch model for training. THis metadata file will be located in the output directory and will be named `metadata.json`. It will also include
information about the tokenizer used to tokenize the data, including the tokenizer's vocabulary and the tokenizer's configuration, as well as the version of Huggingface `tokenizers` and `tatm` used to tokenize the data.

## Loading Tokenized Data with `tatm` for use with PyTorch

Once you have tokenized your data, you can load it into a PyTorch dataset using the `tatm` library. The `tatm` library
provides a pytorch compatible dataset class that can be used to load tokenized data into a PyTorch model for training
([`tatm.data.TatmMemmapDataset`](tatm.data.TatmMemmapDataset)). You can then load the dataset into a PyTorch `DataLoader` and use it to train your
model. The `TatmMemmapDataset` implements the appropriate `__getitem__` and `__len__` methods to be compatible with PyTorch's
`Dataset` API and should support integration with the Pytorch DistrubutedSampler for distributed training.

For an example of what it would look like to load a tokenized dataset into a PyTorch model for training, see the example below:

```python
from tatm.data import get_dataset
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
# TatmMemmapDatasetItem(token_ids=array([    7,    16,     8, ..., 14780,     8,  2537], dtype=uint16), document_ids=array([0, 0, 0, ..., 1, 1, 1], dtype=uint16), document_mask=array([[ True, False, False, ..., False, False, False],
#        [ True,  True, False, ..., False, False, False],
#        [ True,  True,  True, ..., False, False, False],
#        ...,
#        [False, False, False, ...,  True, False, False],
#        [False, False, False, ...,  True,  True, False],
#        [False, False, False, ...,  True,  True,  True]]))
```

Fields in the `TatmMemmapDatasetItem` object include:
- `token_ids`: The tokenized text data
- `document_ids`: The document ids for each token. We use example packing to ease the processing of the data in the LLM. To support document masking, we include the document ids for each token in the dataset.
- `document_mask`: A boolean attention mask that can be used for causal masking of the data. This mask is used to mask out tokens that are not part of the same document as the current token, as well as tokens that should not be considered in the attention calculation for a given token.

For more information on how to use the [`tatm.data.TatmMemmapDataset`](tatm.data.TatmMemmapDataset) class, see the [Data](tatm.data.TatmMemmapDataset) documentation.
