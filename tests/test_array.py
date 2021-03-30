# -*- coding: utf-8 -*-
import numpy as np
import pytest

from sympl import DataArray


def test_array_addition():
    a = DataArray(np.array([1.0, 2.0, 3.0]))
    b = DataArray(np.array([2.0, 1.0, 3.0]))
    result = a + b
    assert (result.values == np.array([3.0, 3.0, 6.0])).all()
    assert len(result.attrs) == 0


def test_array_subtraction():
    a = DataArray(np.array([1.0, 2.0, 3.0]))
    b = DataArray(np.array([2.0, 1.0, 3.0]))
    result = a - b
    assert (result.values == np.array([-1.0, 1.0, 0.0])).all()
    assert len(result.attrs) == 0


def test_array_addition_keeps_left_attr():
    a = DataArray(np.array([1.0, 2.0, 3.0]), attrs={"units": "K"})
    b = DataArray(
        np.array([2.0, 1.0, 3.0]), attrs={"units": "m/s", "foo": "bar"}
    )
    result = a + b
    assert (result.values == np.array([3.0, 3.0, 6.0])).all()
    assert len(result.attrs) == 1
    assert result.attrs["units"] == "K"


def test_array_subtraction_keeps_left_attrs():
    a = DataArray(
        np.array([1.0, 2.0, 3.0]), attrs={"units": "m/s", "foo": "bar"}
    )
    b = DataArray(np.array([2.0, 1.0, 3.0]), attrs={"units": "K"})
    result = a - b
    assert (result.values == np.array([-1.0, 1.0, 0.0])).all()
    assert len(result.attrs) == 2
    assert result.attrs["units"] == "m/s"
    assert result.attrs["foo"] == "bar"


def test_array_unit_conversion_same_units():
    a = DataArray(
        np.array([1.0, 2.0, 3.0]), attrs={"units": "m", "foo": "bar"}
    )
    result = a.to_units("m")
    assert (result.values == np.array([1.0, 2.0, 3.0])).all()
    assert len(result.attrs) == 2
    assert result.attrs["units"] == "m"
    assert result.attrs["foo"] == "bar"


def test_array_unit_conversion_different_units():
    a = DataArray(
        np.array([1.0, 2.0, 3.0]), attrs={"units": "km", "foo": "bar"}
    )
    result = a.to_units("m")
    assert (result.values == np.array([1000.0, 2000.0, 3000.0])).all()
    assert len(result.attrs) == 2
    assert result.attrs["units"] == "m"
    assert result.attrs["foo"] == "bar"
    assert (a.values == np.array([1.0, 2.0, 3.0])).all()
    assert len(a.attrs) == 2
    assert a.attrs["foo"] == "bar"
    assert a.attrs["units"] == "km"


def test_array_unit_conversion_different_units_doesnt_modify_original():
    a = DataArray(
        np.array([1.0, 2.0, 3.0]), attrs={"units": "km", "foo": "bar"}
    )
    a.to_units("m")
    assert (a.values == np.array([1.0, 2.0, 3.0])).all()
    assert len(a.attrs) == 2
    assert a.attrs["foo"] == "bar"
    assert a.attrs["units"] == "km"


if __name__ == "__main__":
    pytest.main([__file__])
