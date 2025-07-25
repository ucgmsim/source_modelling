"""Parsing utilities for space-separated file formats."""

from typing import TextIO


class ParseError(Exception):
    """Error for parsing files in source modelling."""

    pass


def _is_seperator(char: str) -> bool:
    """Check if a character is a space or end of file.

    Parameters
    ----------
    char : str
        The character to check.

    Returns
    -------
    bool
        True if the character is a space or end of file, False otherwise."""
    return char.isspace() or not char


def read_float(handle: TextIO, label: str | None = None) -> float:
    """Read a float from an file.

    Parameters
    ----------
    handle : TextIO
        The file to read from.
    label : str | None
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
    while _is_seperator(cur := handle.read(1)):
        pass
    float_str = cur
    while not _is_seperator(cur := handle.read(1)):
        float_str += cur
    try:
        return float(float_str)
    except ValueError:
        if label:
            raise ParseError(f'Expecting float ({label}), got: "{float_str}"')
        else:
            raise ParseError(f'Expecting float, got: "{float_str}"')


def read_int(handle: TextIO, label: str | None = None) -> int:
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
    while _is_seperator(cur := handle.read(1)):
        pass
    int_str = cur
    while not _is_seperator(cur := handle.read(1)):
        int_str += cur
    try:
        return int(int_str)
    except ValueError:
        if label:
            raise ParseError(f'Expecting int ({label}), got: "{int_str}"')
        else:
            raise ParseError(f'Expecting int, got: "{int_str}"')
