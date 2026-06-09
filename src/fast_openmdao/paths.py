# src/fast_openmdao/paths.py

"""Helpers for reading and writing nested FAST dictionary paths."""

import numpy as np


def normalize_path(path):
    """Return path as tuple components accepted by FAST path helpers.

    Inputs:
        path: Dot-separated string or iterable of dictionary keys/list indexes.

    Outputs:
        Tuple of path components.

    Assumptions:
        Dot-separated strings are for dictionary-like paths. Use an iterable
        when a list index is needed so numeric indexes remain unambiguous.
    """

    if isinstance(path, str):
        return tuple(item for item in path.split(".") if item)

    return tuple(path)


def get_path(data, path):
    """Read a value from a nested dictionary/list path.

    Inputs:
        data: Root object.
        path: Dot-separated string or iterable path.

    Outputs:
        Value stored at the requested path.
    """

    current = data

    for key in normalize_path(path):
        if isinstance(current, list):
            current = current[int(key)]
        else:
            current = current[key]

    return current


def set_path(data, path, value):
    """Write a value into a nested dictionary/list path.

    Inputs:
        data: Mutable root object.
        path: Dot-separated string or iterable path.
        value: Scalar value to assign.

    Outputs:
        The root object, mutated in place and returned for convenience.

    Assumptions:
        The path already exists in the FAST baseline. Missing branches are not
        created because silent structure changes would hide misspelled FAST
        fields during optimization setup.
    """

    components = normalize_path(path)
    current = data

    for key in components[:-1]:
        if isinstance(current, list):
            current = current[int(key)]
        else:
            current = current[key]

    leaf = components[-1]
    scalar = scalar_value(value)

    if isinstance(current, list):
        current[int(leaf)] = scalar
    else:
        current[leaf] = scalar

    return data


def scalar_value(value):
    """Return a plain scalar from OpenMDAO or NumPy scalar containers.

    Inputs:
        value: Scalar, NumPy scalar, or one-element array-like value.

    Outputs:
        Plain Python scalar for assignment into FAST dictionaries.

    Assumptions:
        This first OpenMDAO bridge supports scalar design variables. Vector
        variables should be introduced deliberately once the FAST path contract
        for array-valued quantities is settled.
    """

    array = np.asarray(value)

    if array.shape == ():
        return array.item()

    if array.size == 1:
        return array.reshape(-1)[0].item()

    raise ValueError("FAST OpenMDAO path values must be scalar.")

