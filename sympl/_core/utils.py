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
from inspect import getfullargspec as getargspec
from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING, Union

from sympl._core.data_array import DataArray
from sympl._core.exceptions import InvalidStateError

if TYPE_CHECKING:
    from sympl._core.typingx import Component

try:
    from numba import jit
except ImportError:
    # define a function with the same call signature as jit that does nothing
    def jit(signature_or_function=None, **kwargs):
        if signature_or_function is None:
            return lambda x: x
        else:
            return signature_or_function


def same_list(list1: Sequence, list2: Sequence) -> bool:
    """Returns a boolean indicating whether the items in list1 are the same
    items present in list2 (ignoring order)."""
    return len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]
    )


def update_dict_by_adding_another(dict1: Dict, dict2: Dict) -> None:
    """
    Takes two dictionaries. Add values in dict2 to the values in dict1, if
    present. If not present, create a new value in dict1 equal to the value in
    dict2. Addition is done in-place if the values are
    array-like, to avoid data copying. Units are handled if the values are
    DataArrays with a 'units' attribute.
    """
    for key in dict2.keys():
        if key not in dict1:
            if hasattr(dict2[key], "copy"):
                dict1[key] = dict2[key].copy()
            else:
                dict1[key] = dict2[key]
        else:
            if isinstance(dict1[key], DataArray) and isinstance(
                dict2[key], DataArray
            ):
                if (
                    "units" not in dict1[key].attrs
                    or "units" not in dict2[key].attrs
                ):
                    raise InvalidStateError(
                        "DataArray objects must have units property defined"
                    )
                try:
                    dict1[key] += dict2[key].to_units(
                        dict1[key].attrs["units"]
                    )
                except ValueError:  # dict1[key] is missing a dimension present in dict2[key]
                    dict1[key] = dict1[key] + dict2[key].to_units(
                        dict1[key].attrs["units"]
                    )
            else:
                dict1[key] += dict2[key]  # += is in-place addition operator
    return  # not returning anything emphasizes that this is in-place


def get_component_aliases(
    *args: "Component",
) -> Dict[str, Union[str, Sequence[str]]]:
    """
    Returns aliases for variables in the properties of Components (TendencyComponent,
    DiagnosticComponent, Stepper, and ImplicitTendencyComponent objects).

    If multiple aliases are present for the same variable, the following
    properties have priority in descending order: input, output, diagnostic,
    tendency. If multiple components give different aliases at the same priority
    level, one is chosen arbitrarily.

    Args
    ----
    *args : Component
        Components from which to fetch variable aliases from the input_properties,
        output_properties, diagnostic_properties, and tendency_properties dictionaries

    Returns
    -------
    aliases : dict
        A dictionary mapping quantity names to aliases
    """
    return_dict = {}
    for property_type in (
        "tendency_properties",
        "diagnostic_properties",
        "output_properties",
        "input_properties",
    ):
        for component in args:
            if hasattr(component, property_type):
                component_properties = getattr(component, property_type)
                for name, properties in component_properties.items():
                    if "alias" in properties.keys():
                        return_dict[name] = properties["alias"]
    return return_dict


def get_kwarg_defaults(func: Callable) -> Dict[str, Any]:
    return_dict = {}
    argspec = getargspec(func)
    if argspec.defaults is not None:
        n = len(argspec.args) - 1
        for i, default in enumerate(reversed(argspec.defaults)):
            return_dict[argspec.args[n - i]] = default
    return return_dict
