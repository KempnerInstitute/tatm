import numpy as np

from tatm.data.datasets import _create_document_mask, _get_document_ids


def test_get_document_ids():
    example_data = np.array([42, 71, 1, 987, 666, 809, 1, 42, 1])
    expected_output = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    output = _get_document_ids(example_data, eos_token=1)
    assert np.array_equal(
        output, expected_output
    )  # Check if the output matches the expected output


def test_create_attention_mask():
    example_document_ids = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    expected_mask = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]
    )
    output = _create_document_mask(example_document_ids)
    assert np.all(
        np.array_equal(output, expected_mask)
    )  # Check if the output matches the expected output
