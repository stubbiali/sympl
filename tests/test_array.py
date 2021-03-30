# -*- coding: utf-8 -*-
#
# BSD License
#
# Copyright (c) 2016-2021, Jeremy McGibbon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
#
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
