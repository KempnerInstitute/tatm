from tatm.utils import TatmOptionEnum


def test_tatm_option_enum():
    class TestEnum(TatmOptionEnum):
        TEST1 = "test1"
        TEST2 = "test2"

    assert TestEnum.has_value("test1")
    assert TestEnum.has_value("test2")
    assert not TestEnum.has_value("test3")
