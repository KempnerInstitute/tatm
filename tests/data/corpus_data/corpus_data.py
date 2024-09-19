import datasets

primary_lines = [
    "This is the first primary sentence.",
    "This is the second primary sentence.",
    "This is the third primary sentence.",
    "This is the fourth primary sentence.",
    "This is the fifth primary sentence.",
]

secondary_lines = [
    "This is a test sentence.",
    "Another test sentence.",
    "Yet another test sentence.",
    "This is the fourth test sentence.",
    "Finally, this is the last test sentence.",
]


class CorpusDataConfig(datasets.BuilderConfig):
    """BuilderConfig for CorpusData."""

    def __init__(self, *args, subsets, **kwargs):
        super(CorpusDataConfig, self).__init__(**kwargs)
        self.subsets = subsets


class CorpusData(datasets.GeneratorBasedBuilder):
    """Corpus dataset for testing tatm corpus functionality."""

    DEFAULT_CONFIG_NAME = "default"

    BUILDER_CONFIGS = [
        CorpusDataConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="full dataset",
            subsets=["primary", "secondary"],
        ),
        CorpusDataConfig(
            name="primary",
            version=datasets.Version("1.0.0"),
            description="Primary dataset",
            subsets=["primary"],
        ),
        CorpusDataConfig(
            name="secondary",
            version=datasets.Version("1.0.0"),
            description="Secondary dataset",
            subsets=["secondary"],
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="A dataset for testing tatm corpus functionality.",
            features=datasets.Features({"text": datasets.Value("string")}),
            supervised_keys=None,
            homepage="http://example.com",
            citation="Citation information.",
        )

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"subsets": self.config.subsets},
            ),
        ]

    def _generate_examples(self, subsets):
        """Yields examples."""
        key = 0
        for subset in subsets:
            if subset == "primary":
                for _, line in enumerate(primary_lines):
                    yield key, {"text": line}
                    key += 1
            elif subset == "secondary":
                for _, line in enumerate(secondary_lines):
                    yield key, {"text": line}
                    key += 1
