<!-- .. tatm documentation master file, created by
   sphinx-quickstart on Thu Aug 15 14:58:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. -->

# `tatm`: The Kempner AI Testbed Library

`tatm` (**T**ransformer **A**ssistive **T**estbed **M**odule) is a Python library that provides a set of tools to assist in the development of 
transformer-based (and other architecture) models for research within the Kempner institute. It will provide an interface for accessing and 
manipulating data, training models, and evaluating models. The library is designed to be modular and extensible, allowing for easy integration 
of new models, datasets, and training frameworks.


```{toctree}
:maxdepth: 2
:caption: Introduction:
getting_started.md
configuration.md
```

```{toctree}
:maxdepth: 2
:caption: Examples:
text_dataset.md
metadata.md
dataset_splits.md
```

```{toctree}
:maxdepth: 2
:caption: Administration:
admin_docs/metadata_store_setup.md

```

```{toctree}
:maxdepth: 2
:caption: API Reference:
api_docs/cli.md
api_docs/data_api.md
api_docs/config_api.md
api_docs/tokenizer_api.md
api_docs/metadata_store_api.md
```


