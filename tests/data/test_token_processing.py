import numpy as np

from tatm.data.datasets import _get_document_ids


def test_get_document_ids():
    example_data = np.array([42, 71, 1, 987, 666, 809, 1, 42, 1])
    expected_output = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    output = _get_document_ids(example_data, eos_token=1)
    assert np.array_equal(
        output, expected_output
    )  # Check if the output matches the expected output
