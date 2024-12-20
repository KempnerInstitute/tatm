# Metadata Store Setup

In addition to loading data using paths to specific files and directories, `tatm` also supports integration with a metadata store to allow for the data loading
functions to use semantic names rather than filesystem paths. This is intended to enable users to more easily share data and to enable administrators to manage
data and systems without affecting user code. Example utility scripts are provided to support the setup and management of the metadata store.

## Metadata Store Configuration

### Metadata Store Types

The metadata store can be configured to use different types of backends to store the metadata. The following backends are currently supported:

- JSON Metadata Store: A simple JSON file that stores the metadata in a human-readable format. The file contains a dictionary with the keys being the names of the datasets and the values being the metadata for that dataset formatted as a json object (or a string containing the json for the metadata object). This backend is useful for small-scale deployments or for testing purposes
where users are not expected to need to search or filter the available datasets.

- OpenMetadata Store: A more complex metadata store that uses a database to store the metadata. This backend is useful for larger deployments where users may need to search or filter the available datasets. The OpenMetadata store is built on top of the [OpenMetadata](https://open-metadata.org/) project and supports a wide range of features including search, filtering, and versioning of metadata.

### Configuration File

The type and configuration of the metadata store can be managed through `tatm` configuration files. The `metadata_backend` block in the configuration file can be used to specify the type of metadata store to use and any additional configuration options. See the [metadata backend api docs](../api_docs/metadata_store_api.md) for details on specific arguments required for each backend. The following is an example configuration block for a JSON metadata store:

```yaml
metadata_backend:
  type: json
  args:
    metadata_backend_file: /path/to/metadata.json
```

while the following is an example configuration block for an OpenMetadata store:

```yaml
metadata_backend:
  type: open_metadata
  args:
    address: http://localhost:8080/api
    api_key: <read only api_key>
    data_service_name: tatm-data-service # optional, defaults to 'tatm-data-service'
    
```

As this configuration is likely to be system-wide, it is recommended to place it in the system-wide configuration file at `/etc/tatm/config.yaml`. This will allow all users to access the same metadata store. If that is not feasible, the `TATM_BASE_DIR` or `TATM_BASE_CONFIG` environment variables can be used to specify an alternate configuration file. See the [configuration api docs](../configuration.md) for more information on configuration file locations and environment variables and how they interact.

## Loading metadata into the store

Utility scripts are provided to assist in loading metadata into the metadata store. They can be found in the `scripts/metadata` directory of the `tatm` package. The following scripts are provided:

- `scripts/metadata/create_json_metadata.py`: A script to create a new JSON metadata store file. This script will parse a provided list of structured directories and create a JSON file containing the metadata for each dataset. This script is useful for creating a new JSON metadata store file from a set of directories containing data.
- `scripts/metadata/load_open_metadata_backend.py`: A script to load metadata into an OpenMetadata store. This script will parse a provided list of structured directories and load the metadata for each dataset into the OpenMetadata store. This script makes assumptions based on the names and structure of the directories (see below) and may need to be modified to work with different directory structures.

Note that both scripts assume that all data set names within the metadata store are unique. If a dataset with the same name already exists in the metadata store, the script will overwrite the existing metadata with the new metadata. Care should be taken to ensure that the metadata store is not overwritten unintentionally.

### Testbed Directory Structure

All scripts assume that the data directories are structured in a specific way, termed a "tatm data garden". This structure is as follows:
```
| top_level_directory
|---| Data Context 1 (e.g. "text")
|   |---| Dataset 1 (e.g. "dolma")
|   |   |---| metadata.json # This is the main metadata file for the dataset
|   |   |---| raw # This is the directory containing the raw data, frequently the data as cloned from Huggingface
|   |   |---| tokenized # This is the directory containing the tokenized data, if it has been tokenized
|   |   |   |---| <tokenizer 1> (e.g. "t5-base")
|   |   |   |   |---| <subset descriptor> (e.g. "c4")
|   |   |   |   |   |---| metadata.json # This is the metadata file for the tokenized data
|   |---| Dataset 2 (e.g. "red_pajama")
|   |   |---| ...
|---| Data Context 2 (e.g. "code")
|   |---| ...
```
Names of tokenized datasets are constructed from the subset and their parent dataset, while the "Data Context" is used to tag data with a semantic category. There will be errors
loading the metadata if the directory structure does not match this format. Both scripts currently support multipe directories structured in this way, although there may be a risk
of name collisions if the same dataset is present in multiple directories.

### JSON Metadata Backend

The file at `scripts/metadata/create_json_metadata.py` can be used to create a new JSON metadata store file. This script will parse a provided list of structured directories and create a JSON file containing the metadata for each dataset. It parses the directory structure above, finding all metadata files and adding them to the JSON file. The script can be run with the following command:

```bash
python scripts/metadata/create_json_metadata.py --dirs <comma separated directory list> --output_file /path/to/metadata.json
```

### OpenMetadata Backend

The file at `scripts/metadata/load_open_metadata_backend.py` can be used to load metadata into an OpenMetadata store. This script will parse a provided list of structured directories and load the metadata for each dataset into the OpenMetadata store. It makes assumptions based on the names and structure of the directories and may need to be modified to work with different directory structures. The script can be run with the following command:

```bash
python scripts/metadata/load_open_metadata_backend.py --dirs <comma separated directory list> --host http://example.com --port 8585 --api_key <read only key>
```
The address can also be specified using the `OPEN_METADATA_ADDRESS` environment variable (in the format `http://example.com:8585`), and the api key can be specified using the `OPEN_METADATA_API_KEY` environment variable. This API key should be the key used by the ingestion bot so that the script has permission to update the service

We somewhat hack the OpenMetadata structures to serve our purposes. Currently we represent the testbed as a custom database, creating a top level service to serve as a parent
to the other entities we create. We represent all top level datasets as Databases on that data service, while representing the tokenized datasets and corpuses as schemas
in those databases. This is a bit of a hack, but it allows us to represent the data in a way that is somewhat meaningful to the OpenMetadata system. 

We also tag entities as being tokenized (or not), corpuses (or not), and with the tokenizer used (where applicable), the context of the data, and the content type of the data (i.e. text, image, etc). This allows us to filter and search for data in a meaningful way.