from tatm.data import get_dataset


def test_text_json_dataset_load():
    dataset = get_dataset("tests/data/json_dataset")
    first_item = next(iter(dataset))
    assert first_item["text"] == "hello world"
