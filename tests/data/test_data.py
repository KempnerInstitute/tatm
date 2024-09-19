from tatm.data import get_data


def test_text_json_dataset_load():
    dataset = get_data("tests/data/json_data")
    first_item = next(iter(dataset))
    assert first_item["text"] == "hello world"


def test_corpus_data_load():
    for corpus in ["primary", "secondary"]:
        dataset = get_data(f"tests/data/corpus_data:{corpus}")
        item = 0
        for i in dataset:
            if item == 0:
                if corpus == "primary":
                    assert i["text"] == "This is the first primary sentence."
                else:
                    assert i["text"] == "This is a test sentence."
            item += 1
        assert item == 5


def test_corpus_full_data_load():
    dataset = get_data("tests/data/corpus_data")
    item = 0
    for i in dataset:
        if item == 0:
            assert i["text"] == "This is the first primary sentence."
        item += 1
    assert item == 10
