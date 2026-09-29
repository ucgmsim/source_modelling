import io

import pytest

from source_modelling import parse_utils


def test_read_float_reads_value_separated_by_whitespace():
    handle = io.StringIO("  3.5 -2.25\n")
    assert parse_utils.read_float(handle) == pytest.approx(3.5)
    assert parse_utils.read_float(handle) == pytest.approx(-2.25)


def test_read_float_invalid_value_with_label_reports_label():
    handle = io.StringIO("notafloat ")
    with pytest.raises(
        parse_utils.ParseError,
        match=r'Expecting float \(longitude\), got: "notafloat"',
    ):
        parse_utils.read_float(handle, "longitude")


def test_read_float_invalid_value_without_label_omits_label():
    handle = io.StringIO("notafloat ")
    with pytest.raises(
        parse_utils.ParseError, match=r'Expecting float, got: "notafloat"'
    ) as exc_info:
        parse_utils.read_float(handle)
    assert "longitude" not in str(exc_info.value)


def test_read_int_reads_value_separated_by_whitespace():
    handle = io.StringIO("  42 -7\n")
    assert parse_utils.read_int(handle) == 42
    assert parse_utils.read_int(handle) == -7


def test_read_int_invalid_value_with_label_reports_label():
    handle = io.StringIO("3.5 ")
    with pytest.raises(
        parse_utils.ParseError, match=r'Expecting int \(nx\), got: "3.5"'
    ):
        parse_utils.read_int(handle, "nx")


def test_read_int_invalid_value_without_label_omits_label():
    handle = io.StringIO("3.5 ")
    with pytest.raises(
        parse_utils.ParseError, match=r'Expecting int, got: "3.5"'
    ) as exc_info:
        parse_utils.read_int(handle)
    assert "nx" not in str(exc_info.value)
