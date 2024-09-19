import tatm.data.metadata as metadata


def main():

    # Create metadata for dataset
    dataset_metadata = metadata.TatmDataMetadata(
        name="test_corpus_data",
        dataset_path="tests/data/corpus_data",
        description="A test corpus data set intended for use in testing.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content="text",
        content_field="text",
        corpuses=["primary", "secondary"],
    )
    dataset_metadata.to_yaml("metadata.yaml")


if __name__ == "__main__":
    main()
