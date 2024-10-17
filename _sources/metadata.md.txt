# Dataset Metadata

`tatm` uses metadata files on disk to determine how to load and process data. The metadata file is a JSON file or YAML file that contains information about the dataset, such as the location of the data files, the format of the data, and any other relevant information. These will typically need to be created by administrators or data curators to enable a dataset for use with the library
and allow for users to easily load and process diverse data with a unified API.

## Metadata Fields

The metadata file contains the following fields:
- `name`: The name of the dataset. This field is not currently used by the library, but it can be used to provide a human-readable name for the dataset.
- `dataset_path`: The path to the raw data files for the dataset. Passed to the `datasets` library to load the data.
- `description`: A description of the dataset.
- `date_downloaded`: The date the dataset was downloaded.
- `download_source`: The source from which the dataset was downloaded.
- `data_content`: The type of data in the dataset. This field is used to determine how to process the data. Currently only text data is supported.
- `content_field`: The field that contains the primary data in the dataset. This field is used to determine how to process the data. Assumes that the raw data is 
    stored in a dictionary-like object.
- `corpuses`: A list of corpus names. This field is used to group the data into different corpora. This field is for documentation purposes only and no validation is done at run time
     when loading the data.
- `corpus_separation_strategy`: How the data is separated into corpora. Currently supports `data_dirs` and `configs`. `data_dirs` leads to `tatm` using the `data_dir` field in `datasets.load_dataset` to load the data. `configs` leads to `tatm` using the `config_name` field in `datasets.load_dataset` to load the data. Defaults to `configs` when not set.
- `corpus_data_dir_parent`: The parent directory of the data directories for each corpus. This field is used when the `corpus_separation_strategy` is set to `data_dirs`. This field is prepended to the corpus name to create the full path to the subdirectory for the corpus. Defaults to `None` (aka the top level of the dataset directory) when not set.
- `tokenized_info`: A sub object defining metadata about tokenized data. This field is used to indicate that a dataset is pretokenized and to provide information about the tokenizer used to tokenize the data. This field is used to determine how to load the data and what tokenizer to use when loading the data. This field is optional and only used when the data is tokenized.
    - `tokenizer`: The name of the tokenizer used to tokenize the data. This field is used to determine which tokenizer to use when loading the data. This field is required when the `tokenized_info` field is present. Maps typically to a huggingface tokenizer name.
    - `file_prefix`: The prefix of the tokenized data files. This field is used to determine the file names of the tokenized data files. This field is required when the `tokenized_info` field is present.
    - `dtype`: The data type of the tokenized data. This field is used to determine the data type of the tokenized data when loading the data. This field is optional and defaults to `np.uint16`.
    - `vocab_size`: The size of the vocabulary used by the tokenizer. This field is used to determine the size of the vocabulary when loading the data. This field is optional and defaults to `None`.
    - `tatm_version`: The version of the `tatm` library used to tokenize the data. Provided for reproducibility purposes. This field is optional and defaults to `None`.
    - `tokenizers_version`: The version of the `tokenizers` library used to tokenize the data. Provided for reproducibility purposes. This field is optional and defaults to `None`.


## Creating a Metadata File

### Using the CLI

The `tatm` library provides an interactive CLI tool that can help you create a metadata file. To use this tool, run the following command from the directory where your data is stored:

```bash
tatm data create-metadata
```

The CLI tool will prompt you for information about your data, such as the name of the dataset, the path to the raw data files, and the format of the data. The tool will then create a metadata file that describes the data and how it is stored on disk. The `tatm` library uses this metadata file to load and process the data.

### Using the Python API

The `tatm` library also provides a Python API for creating metadata files. You can use the `tatm.data.TatmDataMetadata` class to create a metadata file programmatically. Here is an example of how to create a metadata file for a text dataset using the Python API:

```python
from tatm.data import TatmDataMetadata

metadata = TatmDataMetadata(
        name="Example Dataset", # Name of the dataset, not currently used by the library
        dataset_path="<ABSOLUTE PATH TO DATA>",
        description="An example text dataset",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content="text", # Type of data in the dataset
        content_field="text", # Assuming that the data presents dictionary-like objects, the field that contains the primary data
        corpuses=["example_corpus", "example_corpus_2"], # List of corpus names. Sub corpora within the dataset, list here is for documentation purposes
        corpus_separation_strategy="data_dirs", # How the data is separated into corpora, currently supports "data_dirs" and "configs"
        corpus_data_dir_parent="data", # Parent directory of the data directories for each corpus. In this example the 
                                       # "example_corpus" data is stored in "data/example_corpus" within the dataset directory
    )
metadata.to_yaml("metadata.yaml")
```



