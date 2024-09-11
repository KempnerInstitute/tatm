from tatm.data import get_data


def test_text_json_dataset_load():
    dataset = get_data("tests/data/json_data")
    first_item = next(iter(dataset))
    assert first_item["text"] == "hello world"
