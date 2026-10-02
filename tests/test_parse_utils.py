import io
import multiprocessing

import pytest

from source_modelling import parse_utils


def _run_read_float_on_blank_stream(queue: multiprocessing.Queue) -> None:
    try:
        parse_utils.read_float(io.StringIO("  "), "x")
        queue.put("returned")
    except parse_utils.ParseError as exc:
        queue.put(str(exc))


def test_read_float_on_exhausted_stream_raises_instead_of_hanging():
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_read_float_on_blank_stream, args=(queue,)
    )
    process.start()
    process.join(timeout=5)
    still_running = process.is_alive()
    if still_running:
        process.terminate()
        process.join()
    assert not still_running, "read_float on exhausted stream: still running after 5s"
    assert "Unexpected end of file" in queue.get()


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
