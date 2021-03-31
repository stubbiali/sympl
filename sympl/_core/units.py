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
import functools

import pint


class UnitRegistry(pint.UnitRegistry):
    @functools.lru_cache
    def __call__(self, input_string, **kwargs):
        return super(UnitRegistry, self).__call__(
            input_string.replace("%", "percent").replace("°", "degree"),
            **kwargs
        )


unit_registry = UnitRegistry()
unit_registry.define(
    "degrees_north = degree_north = degree_N = degrees_N = degreeN = degreesN"
)
unit_registry.define(
    "degrees_east = degree_east = degree_E = degrees_E = degreeE = degreesE"
)
unit_registry.define("percent = 0.01*count = %")


def units_are_compatible(unit1, unit2):
    """
    Determine whether a unit can be converted to another unit.

    Parameters
    ----------
    unit1 : str
    unit2 : str

    Returns
    -------
    units_are_compatible : bool
        True if the first unit can be converted to the second unit.
    """
    try:
        unit_registry(unit1).to(unit2)
        return True
    except pint.errors.DimensionalityError:
        return False


def units_are_same(unit1, unit2):
    """
    Compare two unit strings for equality.

    Parameters
    ----------
    unit1 : str
    unit2 : str

    Returns
    -------
    units_are_same : bool
        True if the two input unit strings represent the same unit.
    """
    return unit_registry(unit1) == unit_registry(unit2)


def clean_units(unit_string):
    return str(unit_registry(unit_string).to_base_units().units)


def is_valid_unit(unit_string):
    """Returns True if the unit string is recognized, and False otherwise."""
    unit_string = unit_string.replace("%", "percent").replace("°", "degree")
    try:
        unit_registry(unit_string)
    except pint.UndefinedUnitError:
        return False
    else:
        return True


def data_array_to_units(value, units):
    if not hasattr(value, "attrs") or "units" not in value.attrs:
        raise TypeError(
            "Cannot retrieve units from type {}".format(type(value))
        )
    elif unit_registry(value.attrs["units"]) != unit_registry(units):
        out = value.copy()
        out.data[...] = (
            unit_registry.convert(1, value.attrs["units"], units) * value.data
        )
        out.attrs["units"] = units
        value = out
    return value


def from_unit_to_another(value, original_units, new_units):
    return (unit_registry(original_units) * value).to(new_units).magnitude


def get_name_with_incompatible_units(properties1, properties2):
    """
    If there are any keys shared by the two properties
    dictionaries which indicate units that are incompatible with one another,
    this returns such a key. Otherwise returns None.
    """
    for name in set(properties1.keys()).intersection(properties2.keys()):
        if not units_are_compatible(
            properties1[name]["units"], properties2[name]["units"]
        ):
            return name
    return None


def get_tendency_name_with_incompatible_units(
    input_properties, tendency_properties
):
    """
    Returns False if there are any keys shared by the two properties
    dictionaries which indicate units that are incompatible with one another,
    and True otherwise (if there are no conflicting unit specifications).
    """
    for name in set(input_properties.keys()).intersection(
        tendency_properties.keys()
    ):
        if input_properties[name]["units"] == "":
            expected_tendency_units = "s^-1"
        else:
            expected_tendency_units = input_properties[name]["units"] + " s^-1"
        if not units_are_compatible(
            expected_tendency_units, tendency_properties[name]["units"]
        ):
            return name
    return None
