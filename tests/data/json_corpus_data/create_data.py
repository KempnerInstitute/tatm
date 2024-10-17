import json
import os

from tatm.data.metadata import TatmDataMetadata

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

if __name__ == "__main__":
    os.makedirs("data/primary", exist_ok=True)
    with open("data/primary/primary_text_dataset.json", "w") as f:
        for line in primary_lines:
            f.write(json.dumps({"text": line}) + "\n")

    os.makedirs("data/secondary", exist_ok=True)
    with open("data/secondary/secondary_text_dataset.json", "w") as f:
        for line in secondary_lines:
            f.write(json.dumps({"text": line}) + "\n")

    metadata = TatmDataMetadata(
        name="json_corpus_data",
        dataset_path="tests/data/json_corpus_data",
        description="A test dataset with primary and secondary corpuses.",
        data_content="text",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        content_field="text",
        corpuses=["primary", "secondary"],
        corpus_separation_strategy="data_dirs",
        tokenized_info=None,
    )

    metadata.to_yaml("metadata.yaml")
