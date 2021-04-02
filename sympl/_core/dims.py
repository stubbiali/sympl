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
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple

from sympl._core.checks import ensure_properties_have_dims_and_units
from sympl._core.exceptions import (
    InvalidPropertyDictError,
    InvalidStateError,
    NoMatchForDirectionError,
)

if TYPE_CHECKING:
    from sympl._core.typing import (
        DataArray,
        DataArrayDict,
        NDArrayLikeDict,
        PropertyDict,
    )


def fill_dims_wildcard(
    out_dims: Sequence[str],
    dim_lengths: Dict[str, int],
    wildcard_names: Sequence[str],
    expand_wildcard: bool = True,
) -> Tuple[List[str], List[int]]:
    i_wildcard = out_dims.index("*")
    target_shape = []
    out_dims_without_wildcard = []

    for i, out_dim in enumerate(out_dims):
        if i == i_wildcard and expand_wildcard:
            target_shape += [dim_lengths[n] for n in wildcard_names]
            out_dims_without_wildcard += wildcard_names
        elif i == i_wildcard and not expand_wildcard:
            target_shape.append(
                np.product([dim_lengths[n] for n in wildcard_names])
            )
        else:
            target_shape.append(dim_lengths[out_dim])
            out_dims_without_wildcard.append(out_dim)

    return out_dims_without_wildcard, target_shape


def get_alias_or_name(
    name: str,
    output_properties: "PropertyDict",
    input_properties: "PropertyDict",
) -> str:
    if "alias" in output_properties[name]:
        return output_properties[name]["alias"]
    elif name in input_properties and "alias" in input_properties[name]:
        return input_properties[name]["alias"]
    else:
        return name


def get_final_shape(
    data_array: "DataArray",
    out_dims: Sequence[str],
    direction_to_names: Dict[str, Sequence[str]],
) -> List[int]:
    """
    Determine the final shape that data_array must be reshaped to in order to
    have one axis for each of the out_dims (for instance, combining all
    axes collected by the '*' direction).
    """
    final_shape = []
    for direction in out_dims:
        if len(direction_to_names[direction]) == 0:
            final_shape.append(1)
        else:
            # determine shape once dimensions for direction (usually '*')
            # are combined
            final_shape.append(
                np.product(
                    [
                        len(data_array.coords[name])
                        for name in direction_to_names[direction]
                    ]
                )
            )
    return final_shape


def get_array_dim_names(
    data_array: "DataArray",
    out_dims: Sequence[str],
    dim_names: Dict[str, str],
    *,
    enable_checks: bool = True
) -> Dict[str, List[str]]:
    """
    Parameters
    ----------
    data_array : DataArray
    out_dims : iterable
        directions in dim_names that should be included in the output,
        in the order they should be included
    dim_names : dict
        a mapping from directions to dimension names that fall under that
        direction wildcard.

    Returns
    -------
    array_dim_names : dict
        A mapping from directions included in out_dims to the directions
        present in data_array that correspond to those directions
    """
    array_dim_names = {}

    for direction in out_dims:
        if direction != "*":
            matching_dims = set(data_array.dims).intersection(
                dim_names[direction]
            )
            # must ensure matching dims are in right order
            array_dim_names[direction] = []
            for dim in data_array.dims:
                if dim in matching_dims:
                    array_dim_names[direction].append(dim)

            if enable_checks:
                if (
                    direction not in ("x", "y", "z", "*")
                    and len(array_dim_names[direction]) == 0
                ):
                    raise NoMatchForDirectionError(direction)

    if "*" in out_dims:
        matching_dims = set(data_array.dims).difference(
            *array_dim_names.values()
        )
        array_dim_names["*"] = []
        for dim in data_array.dims:
            if dim in matching_dims:
                array_dim_names["*"].append(dim)

    return array_dim_names


def get_slices_and_placeholder_nones(
    data_array: "DataArray",
    out_dims: Sequence[str],
    direction_to_names: Dict[str, Sequence[str]],
    *,
    enable_checks: bool = True
) -> List[Optional[slice]]:
    """
    Takes in a DataArray, a desired ordering of output directions, and
    a dictionary mapping those directions to a list of names corresponding to
    those directions. Returns a list with the same ordering as out_dims that
    contains slices for out_dims that have corresponding names (as many slices
    as names, and spanning the entire dimension named), and None for out_dims
    without corresponding names.

    This returned list can be used to create length-1 axes for the dimensions
    that currently have no corresponding names in data_array.
    """
    slices_or_none = []

    for direction in out_dims:
        if enable_checks:
            if direction != "*" and len(direction_to_names[direction]) > 1:
                raise ValueError(
                    "DataArray has multiple dimensions for a single direction"
                )

        if len(direction_to_names[direction]) == 0:
            slices_or_none.append(None)
        else:
            for name in direction_to_names[direction]:
                slices_or_none.append(slice(0, len(data_array.coords[name])))

    return slices_or_none


def get_target_dimension_order(
    out_dims: Sequence[str], direction_to_names: Dict[str, Sequence[str]]
) -> List[str]:
    """
    Takes in an iterable of directions ('x', 'y', 'z', or '*') and a dictionary
    mapping those directions to a list of names corresponding to those
    directions. Returns a list of names in the same order as in out_dims,
    preserving the order within direction_to_names for each direction.
    """
    target_dimension_order = []
    for direction in out_dims:
        target_dimension_order.extend(direction_to_names[direction])
    return target_dimension_order


def get_wildcard_names_and_dim_lengths(
    state: "DataArrayDict",
    property_dictionary: "PropertyDict",
    *,
    enable_checks: bool = True
) -> Tuple[Optional[Tuple[str]], Dict[str, int]]:
    wildcard_names = set()
    dim_lengths = {}

    # loop to get the set of names matching "*" (wildcard names)
    for quantity_name, properties in property_dictionary.items():
        if enable_checks:
            ensure_properties_have_dims_and_units(properties, quantity_name)

        for dim_name, length in zip(
            state[quantity_name].dims, state[quantity_name].shape
        ):
            out_length = dim_lengths.setdefault(dim_name, length)
            if out_length != length:
                raise InvalidStateError(
                    f"Dimension {dim_name} conflicting lengths {out_length} "
                    f"and {length} in different state quantities."
                )

        new_wildcard_names = [
            dim
            for dim in state[quantity_name].dims
            if dim not in properties["dims"]
        ]
        if enable_checks:
            if len(new_wildcard_names) > 0 and "*" not in properties["dims"]:
                raise InvalidStateError(
                    f"Quantity {quantity_name} has unexpected dimensions "
                    f"{new_wildcard_names}."
                )
        wildcard_names.update(new_wildcard_names)

    wildcard_names = tuple(wildcard_names) if len(wildcard_names) > 0 else None

    return wildcard_names, dim_lengths


def extract_output_dims_properties(
    output_properties: "PropertyDict",
    input_properties: "PropertyDict",
    ignore_names: Sequence[str],
) -> Dict[str, Sequence[str]]:
    out = {}

    for name, properties in output_properties.items():
        if name in ignore_names:
            continue
        elif "dims" in properties:
            out[name] = properties["dims"]
        elif name not in input_properties:
            raise InvalidPropertyDictError(
                f"Output dims must be specified for {name} in properties."
            )
        elif "dims" not in input_properties[name]:
            raise InvalidPropertyDictError(
                f"Input dims must be specified for {name} in properties."
            )
        else:
            out[name] = input_properties[name]["dims"]

    return out


def get_dim_lengths_from_raw_input(
    raw_input: "NDArrayLikeDict", input_properties: "PropertyDict"
) -> Dict[str, int]:
    dim_lengths = {}
    for name, properties in input_properties.items():
        if properties.get("tracer", False):
            continue
        if "alias" in properties:
            name = properties["alias"]
        for i, dim_name in enumerate(properties["dims"]):
            if dim_name in dim_lengths:
                if raw_input[name].shape[i] != dim_lengths[dim_name]:
                    raise InvalidStateError(
                        f"Dimension name {dim_name} has differing lengths on "
                        f"different inputs."
                    )
            else:
                dim_lengths[dim_name] = raw_input[name].shape[i]
    return dim_lengths
