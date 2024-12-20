# Package Configuration

`tatm` uses a configuration file to manage user and system level settings. The configuration file is a YAML file that is loaded at runtime and used to set the default values for the various options in the library. The configuration file is loaded from the following locations in order they are loaded, with later files taking precedence:
- `/etc/tatm/config.yaml`: The system-wide configuration file. This file is used to set system-wide defaults for all users of the system. It is recommended that this file be used to set the default values for the metadata store and other system-wide settings.
- `$TATM_BASE_DIR/config/config.yaml`: A secondary system-wide configuration file. This should be used if adminstrators don't have access to the system-wide configuration file or they want to override specific settings.
- `$TATM_BASE_CONFIG`: An environment variable that can be used to specify an alternate configuration file. This can be used to set user-specific configuration options or override system-wide options.
- CLI provided configuration files: Configuration files can be provided to the CLI using the `--config` option. These files will be merged with the system-wide configuration files and the environment variable configuration file.
- CLI provided dotlist configuration options: Configuration options can be provided to the CLI using the `--config` option. These options will be merged with the system-wide configuration files and the environment variable configuration file, taking ultimate precedence.

## Examples and details
- For details on specific configuration blocks and options, see the [configuration api docs](api_docs/config_api.md).

- For an example of User level configuration being used to specify how to interact with slurm, see [](text_dataset.md).

- For an example of how to use configuration file to set the metadata store, see the [metadata store setup documentation](admin_docs/metadata_store_setup.md).

