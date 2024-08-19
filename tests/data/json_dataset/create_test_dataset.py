import json

from tatm.data import DatasetMetadata

lines = [
    "hello world",
    "is it me you're looking for?",
    "I can see it in your eyes",
    "I can see it in your smile",
    "You're all I've ever wanted",
    "And my arms are open wide",
]


def main():
    with open("text_datset.json", "w") as f:
        for line in lines:
            f.write(json.dumps({"text": line}) + "\n")

    metadata = DatasetMetadata(
        name="test",
        dataset_path="tests/data/test_json_dataset",
        description="A test json dataset intended for use in testing.",
        date_downloaded="2021-01-01",
        download_source="http://example.com",
        data_content="text",
        content_field="text",
    )
    metadata.to_yaml("metadata.yaml")


if __name__ == "__main__":
    main()
