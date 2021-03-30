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
import pytest

from sympl import DataArray, get_constant, set_constant
from sympl._core.constants import constants
from sympl._core.units import is_valid_unit


def test_constants_are_dataarray():
    for constant_name, value in constants.items():
        assert isinstance(value, DataArray), constant_name


def test_constants_have_valid_units():
    for constant_name, value in constants.items():
        assert "units" in value.attrs, constant_name
        assert is_valid_unit(value.attrs["units"]), constant_name


def test_setting_existing_constant():

    set_constant("seconds_per_day", 100000, "seconds/day")
    new_constant = get_constant("seconds_per_day", units="seconds/day")
    assert new_constant == 100000


def test_setting_new_constant():

    set_constant("my_own_constant", 10.0, "W m^-1 degK^-1")
    new_constant = get_constant("my_own_constant", units="W m^-1 degK^-1")
    assert new_constant == 10.0


def test_converting_existing_constant():
    g_m_per_second = get_constant("gravitational_acceleration", "m s^-2")
    g_km_per_second = get_constant("gravitational_acceleration", "km s^-2")
    assert g_km_per_second == g_m_per_second * 0.001


def test_setting_wrong_units():

    with pytest.raises(ValueError) as excinfo:
        set_constant("abcd", 100, "Wii")

    assert "valid unit" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__])
