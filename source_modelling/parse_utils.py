"""Parsing utilities for space-separated file formats."""

from typing import Optional, TextIO


class ParseError(Exception):
    pass


def read_float(handle: TextIO, label: Optional[str] = None) -> float:
    """Read a float from an file.

    Parameters
    ----------
    handle : TextIO
        The file to read from.
    label : Optional[str]
        A human friendly label for the floating point (for debugging
        purposes), or None for no label. Defaults to None.

    Raises
    ------
    ParseError
        If there is an error reading the float value from the file.

    Returns
    -------
    float
        The float read from the file.
    """
    while (cur := handle.read(1)).isspace():
        pass
    float_str = cur
    while not (cur := handle.read(1)).isspace():
        float_str += cur
    try:
        return float(float_str)
    except ValueError:
        if label:
            raise ParseError(f'Expecting float ({label}), got: "{float_str}"')
        else:
            raise ParseError(f'Expecting float, got: "{float_str}"')


def read_int(handle: TextIO, label: Optional[str] = None) -> int:
    """Read a int from an file.

    Parameters
    ----------
    handle : TextIO
        The file to read from.
    label : str
        Human readable string label for identifying the value in an error
        message.

    Raises
    ------
    ParseError
        If there is an error reading the int value from the file.

    Returns
    -------
    int
        The int read from the file.
    """
    while (cur := handle.read(1)).isspace():
        pass
    int_str = cur
    while not (cur := handle.read(1)).isspace():
        int_str += cur
    try:
        return int(int_str)
    except ValueError:
        if label:
            raise ParseError(f'Expecting int ({label}), got: "{int_str}"')
        else:
            raise ParseError(f'Expecting int, got: "{int_str}"')
