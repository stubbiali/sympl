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
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sympl._core.typing import DataArray, NDArrayLike, PropertyDict


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


def units_are_compatible(unit1: str, unit2: str) -> bool:
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


def units_are_same(unit1: str, unit2: str) -> bool:
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


def clean_units(unit: str) -> str:
    return str(unit_registry(unit).to_base_units().units)


def is_valid_unit(unit: str) -> bool:
    """Returns True if the unit string is recognized, and False otherwise."""
    unit = unit.replace("%", "percent").replace("°", "degree")
    try:
        unit_registry(unit)
    except pint.UndefinedUnitError:
        return False
    else:
        return True


def data_array_to_units(
    data_array: "DataArray", units: str, *, enable_checks: bool = True
) -> "DataArray":
    if enable_checks:
        if not hasattr(data_array, "attrs") or "units" not in data_array.attrs:
            raise TypeError(
                "Cannot retrieve units from type {}".format(type(data_array))
            )

    if unit_registry(data_array.attrs["units"]) != unit_registry(units):
        out = data_array.copy()
        out.data[...] = (
            unit_registry.convert(1, data_array.attrs["units"], units)
            * data_array.data
        )
        out.attrs["units"] = units
        data_array = out

    return data_array


def from_unit_to_another(
    array: "NDArrayLike", original_units: str, new_units: str
) -> "NDArrayLike":
    return (unit_registry(original_units) * array).to(new_units).magnitude


def get_name_with_incompatible_units(
    properties1: "PropertyDict", properties2: "PropertyDict"
) -> Optional[str]:
    """
    If there are any keys shared by the two properties
    dictionaries which indicate units that are incompatible with one another,
    this returns such a key. Otherwise returns None.
    """
    for name in properties1:
        if name in properties2 and not units_are_compatible(
            properties1[name]["units"], properties2[name]["units"]
        ):
            return name
    return None


def get_tendency_name_with_incompatible_units(
    input_properties: "PropertyDict", tendency_properties: "PropertyDict"
) -> Optional[str]:
    """
    Returns False if there are any keys shared by the two properties
    dictionaries which indicate units that are incompatible with one another,
    and True otherwise (if there are no conflicting unit specifications).
    """
    for name in input_properties:
        if name in tendency_properties:
            if input_properties[name]["units"] == "":
                expected_tendency_units = "s^-1"
            else:
                expected_tendency_units = (
                    input_properties[name]["units"] + " s^-1"
                )
            if not units_are_compatible(
                expected_tendency_units, tendency_properties[name]["units"]
            ):
                return name
    return None
