# `tatm` CLI Reference

The `tatm` library includes a command-line interface (CLI) that provides a set of tools for interacting with the library. The CLI is designed to be
easy to use and to provide a consistent interface for common tasks. The CLI is built using the `click` library, which provides a simple and
intuitive way to define command-line interfaces in Python. All commands include a help message that describes the command and provides detailed
information on the options that can be accessed by passing a `--help` flag. This document provides an overview of the commands available in the
`tatm` CLI and how to use them.

## Organization

The `tatm` CLI consists of both top level commands and (where needed) subcommands. This documment is organized following the structure of the CLI.

## Installation

Installing the `tatm` library will automatically install the CLI to your environment.

**DETAILED INSTALLATION INSTRUCTIONS TO BE ADDED**

## Commands

### `tatm run`

```
Usage: tatm run [OPTIONS] [WRAPPED_COMMAND]...

  The `tatm run` command is used to wrap other tatm commands and run them in a
  specified compute environment. It uses the configuration files and options
  to determine how to run the command. `tatm run` will use a template submit
  script along with your specified environment to run the command. It will
  then submit the job to the compute environment. If you do not want to submit
  the job, you can use the `--no-submit` flag. The generated submit script
  will be placed in the current working directory. The submit script will be
  named `tatm_{command}.submit` where `{command}` is the command you are
  running unless you specify a different name with the `--submit-script`
  option.

  If the script is not submitted the command will print the submit command to
  the console.

  The WRAPPED_COMMAND argument is the command that will be run. Currently only
  wrapping ray based commands on slurm is supported.

Options:
  --config, --conf TEXT           Path to configuration file or specific
                                  configuration settings. First, all config
                                  files are merged, then all config options
                                  are merged. Command line options override
                                  config file options. CLI config options
                                  should be in the format
                                  `field.subfield=value`. For list types, use
                                  dotlist notation (e.g.
                                  `field.subfield=[1,2,3]`). Note that this
                                  will override any list values in the config
                                  file. Also note that this may cause issues
                                  with your shell, so be sure to quote the
                                  entire argument.
  -N, --nodes INTEGER             Number of nodes to use for wrapped command.
  -c, --cpus-per-task, --cpus INTEGER
                                  Number of CPUs to use per task.
  --submit-script TEXT            Path to submit script to create.
  --time-limit TEXT               Time limit for the job.
  --memory, --mem TEXT            Memory to allocate for the job.
  --gpus-per-node TEXT            Number of GPUs to use per node.
  --constraints TEXT              Constraints for the job.
  -o, --log-file TEXT             Log file for the job.
  -e, --error-file TEXT           Error file for the job.
  --submit / --no-submit          Submit the job after creating the submit
                                  script. Set to False to only create the
                                  submit script.
  --help                          Show this message and exit.
```

Example usage:

```bash
tatm run --conf slurm.partition=example --conf slurm.account=example -N 4 -c 40 tokenize --output-dir /$OUTPUT_DIR/test_tatm_out/tokenize -v /$DATADIR/redpajama-v1/
```
will run the `tokenize` command with the specified configuration file create a 4 node, 160 CPU ray cluster and tokenize the dataset located at `/DATADIR/redpajama-v1/` and output the tokenized data to `/OUTPUT_DIR/test_tatm_out/tokenize` using 158 workers (2 CPUS are reserved for the writer and reader processes).

The submission script will be created in the current working directory and will be named `tatm_tokenize.submit`. The executed command will be
```
/usr/bin/sbatch --nodes 4 --cpus-per-task 40 --mem 40G --time 1-00:00:00 --partition example --account example --job-name tatm_tokenize --output tatm_tokenize.out $PWD/tatm_tokenize_job.submit
```

### `tatm data`

The data command provides a set of sub commands for with the data layer functionality of the library.

#### `tatm data create-metadata`

The `create-metadata` command kicks off an interactive process to create a metadata file for a dataset. This metadata file
encodes information about the dataset, such as the location of the data files, the format of the data, and any other relevant
information. The metadata file is used by the library to load and process the data.

```
Usage: tatm data create-metadata [OPTIONS]

Options:
  --help  Show this message and exit.
```

### `tatm tokenize`

```
Usage: tatm tokenize [OPTIONS] [DATASETS]...

  Tokenize a dataset using the tatm ray based tokenization engine. If running
  in a cluster environment, it is recommended to use this command in
  conjunction with the `run` command to submit the tokenization job to the
  cluster.

  This command will tokenize the input datasets using the specified tokenizer
  and output the tokenized data to the specified output directory as  a series
  of binary files. The number of workers to use for tokenization can be
  specified using the `--num-workers` option. If not specified, the number of
  workers will be determined by the number of available CPUs  to the ray
  cluster.

  Arguments: DATASETS: Paths to the datasets to tokenize. This command can
  accept multiple datasets to tokenize. All datasets are expected to have a
  tatm metadata file associated with them.

Options:
  --num-workers TEXT  Number of workers to use for tokenization
  --tokenizer TEXT    Tokenizer to use for tokenization
  --output-dir PATH   Output directory for tokenized data
  -v, --verbose       Enable verbose logging
  --file-prefix TEXT  Prefix for tokenized files
  --token-dtype TEXT     Numpy data type for tokenized files
  --help              Show this message and exit.
```