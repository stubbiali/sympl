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
from typing import Iterable, List, Optional, TYPE_CHECKING

from sympl._core.exceptions import InvalidPropertyDictError
from sympl._core.tracers import get_tracer_input_properties
from sympl._core.units import units_are_compatible

if TYPE_CHECKING:
    from sympl._core.typingx import Component, PropertyDict


def combine_dims(dims1: Iterable[str], dims2: Iterable[str]) -> List[str]:
    """
    Takes in two dims specifications and returns a single specification that
    satisfies both, if possible. Raises an InvalidPropertyDictError if not.

    Parameters
    ----------
    dims1 : iterable of str
    dims2 : iterable of str

    Returns
    -------
    dims : iterable of str

    Raises
    ------
    InvalidPropertyDictError
        If the two dims specifications cannot be combined
    """
    if dims1 == dims2:
        return list(dims1)

    dims1 = set(dims1)
    dims1_wildcard = "*" in dims1
    dims1.discard("*")
    dims2 = set(dims2)
    dims2_wildcard = "*" in dims2
    dims2.discard("*")

    unmatched_dims = dims1.union(dims2)
    shared_dims = dims1.intersection(dims2)
    dims_out = []

    if dims1_wildcard and dims2_wildcard:
        dims_out.append("*")
    elif not dims1_wildcard and not dims2_wildcard:
        if shared_dims != dims1 or shared_dims != dims2:
            raise InvalidPropertyDictError(
                f"dims {dims1} and {dims2} are incompatible."
            )
    elif dims1_wildcard:
        if shared_dims != dims2:
            raise InvalidPropertyDictError(
                f"dims {dims1} and {dims2} are incompatible."
            )
    elif dims2_wildcard:
        if shared_dims != dims1:
            raise InvalidPropertyDictError(
                f"dims {dims1} and {dims2} are incompatible."
            )

    dims_out.extend(unmatched_dims)

    return dims_out


def combine_component_properties(
    component_list: Iterable["Component"],
    property_name: str,
    input_properties: Optional["PropertyDict"] = None,
) -> "PropertyDict":
    property_list = []
    for component in component_list:
        property_list.append(getattr(component, property_name))
        if property_name == "input_properties" and getattr(
            component, "uses_tracers", False
        ):
            tracer_dims = list(component.tracer_dims)
            if "tracer" not in tracer_dims:
                raise InvalidPropertyDictError(
                    "tracer_dims must include a 'tracer' dimension indicating "
                    "tracer number"
                )
            tracer_dims.remove("tracer")
            property_list.append(
                get_tracer_input_properties(
                    getattr(component, "prepend_tracers", ()), tracer_dims
                )
            )
    return combine_properties(property_list, input_properties)


def combine_properties(
    property_list: Iterable["PropertyDict"],
    input_properties: Optional["PropertyDict"] = None,
) -> "PropertyDict":
    input_properties = input_properties or {}
    return_dict = {}

    for property_dict in property_list:
        for name, properties in property_dict.items():
            if name not in return_dict:
                return_dict[name] = properties.copy()
                if "dims" not in properties:
                    if (
                        name in input_properties
                        and "dims" in input_properties[name]
                    ):
                        return_dict[name]["dims"] = input_properties[name][
                            "dims"
                        ]
                    else:
                        raise InvalidPropertyDictError()
            elif not units_are_compatible(
                properties["units"], return_dict[name]["units"]
            ):
                raise InvalidPropertyDictError(
                    f"Cannot combine components with incompatible units "
                    f"{return_dict[name]['units']} and {properties['units']} "
                    f"for quantity {name}."
                )
            else:
                if "dims" in properties:
                    new_dims = properties["dims"]
                elif (
                    name in input_properties
                    and "dims" in input_properties[name]
                ):
                    new_dims = input_properties[name]["dims"]
                else:
                    raise InvalidPropertyDictError()

                try:
                    dims = combine_dims(return_dict[name]["dims"], new_dims)
                    return_dict[name]["dims"] = dims
                except InvalidPropertyDictError as err:
                    raise InvalidPropertyDictError(
                        f"Incompatibility between dims of quantity {name}: "
                        f"{err.args[0]}"
                    )

    return return_dict
