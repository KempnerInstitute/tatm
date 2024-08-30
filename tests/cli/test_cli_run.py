from tatm.cli.run import parse_config_opts


def test_parse_config_opts():
    opts = ["config1", "config2", "field.subfield=value"]
    files, overrides = parse_config_opts(opts, validate=False)
    assert files == ["config1", "config2"]
    assert overrides == ["field.subfield=value"]
