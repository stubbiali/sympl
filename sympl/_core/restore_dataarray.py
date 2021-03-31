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
from sympl._core.checks import ensure_values_are_arrays, check_array_shape
from sympl._core.data_array import DataArray
from sympl._core.exceptions import InvalidPropertyDictError
from sympl._core.wildcard import (
    expand_array_wildcard_dims,
    fill_dims_wildcard,
    get_wildcard_matches_and_dim_lengths,
)


def get_alias_or_name(name, output_properties, input_properties):
    if "alias" in output_properties[name].keys():
        raw_name = output_properties[name]["alias"]
    elif (
        name in input_properties.keys()
        and "alias" in input_properties[name].keys()
    ):
        raw_name = input_properties[name]["alias"]
    else:
        raw_name = name
    return raw_name


def restore_data_arrays_with_properties(
    raw_arrays,
    output_properties,
    input_state,
    input_properties,
    ignore_names=None,
    ignore_missing=False,
):
    """
    Parameters
    ----------
    raw_arrays : dict
        A dictionary whose keys are quantity names and values are numpy arrays
        containing the data for those quantities.
    output_properties : dict
        A dictionary whose keys are quantity names and values are dictionaries
        with properties for those quantities. The property "dims" must be
        present for each quantity not also present in input_properties. All
        other properties are included as attributes on the output DataArray
        for that quantity, including "units" which is required.
    input_state : dict
        A state dictionary that was used as input to a component for which
        DataArrays are being restored.
    input_properties : dict
        A dictionary whose keys are quantity names and values are dictionaries
        with input properties for those quantities. The property "dims" must be
        present, indicating the dimensions that the quantity was transformed to
        when taken as input to a component.
    ignore_names : iterable of str, optional
        Names to ignore when encountered in output_properties, will not be
        included in the returned dictionary.
    ignore_missing : bool, optional
        If True, ignore any values in output_properties not present in
        raw_arrays rather than raising an exception. Default is False.

    Returns
    -------
    out_dict : dict
        A dictionary whose keys are quantities and values are DataArrays
        corresponding to those quantities, with data, shapes and attributes
        determined from the inputs to this function.

    Raises
    ------
    InvalidPropertyDictError
        When an output property is specified to have dims_like an input
        property, but the arrays for the two properties have incompatible
        shapes.
    """
    raw_arrays = raw_arrays.copy()
    if ignore_names is None:
        ignore_names = []
    if ignore_missing:
        ignore_names = (
            set(output_properties.keys())
            .difference(raw_arrays.keys())
            .union(ignore_names)
        )
    wildcard_names, dim_lengths = get_wildcard_matches_and_dim_lengths(
        input_state, input_properties
    )
    ensure_values_are_arrays(raw_arrays)
    dims_from_out_properties = extract_output_dims_properties(
        output_properties, input_properties, ignore_names
    )
    out_dict = {}
    for name, out_dims in dims_from_out_properties.items():
        if name in ignore_names:
            continue
        raw_name = get_alias_or_name(name, output_properties, input_properties)
        if "*" in out_dims:
            for dim_name, length in zip(out_dims, raw_arrays[raw_name].shape):
                if dim_name not in dim_lengths and dim_name != "*":
                    dim_lengths[dim_name] = length
            out_dims_without_wildcard, target_shape = fill_dims_wildcard(
                out_dims, dim_lengths, wildcard_names
            )
            out_array = expand_array_wildcard_dims(
                raw_arrays[raw_name], target_shape, name, out_dims
            )
        else:
            check_array_shape(
                out_dims, raw_arrays[raw_name], name, dim_lengths
            )
            out_dims_without_wildcard = out_dims
            out_array = raw_arrays[raw_name]
        out_dict[name] = DataArray(
            out_array,
            dims=out_dims_without_wildcard,
            attrs={"units": output_properties[name]["units"]},
        )
    return out_dict


def extract_output_dims_properties(
    output_properties, input_properties, ignore_names
):
    return_array = {}
    for name, properties in output_properties.items():
        if name in ignore_names:
            continue
        elif "dims" in properties.keys():
            return_array[name] = properties["dims"]
        elif name not in input_properties.keys():
            raise InvalidPropertyDictError(
                "Output dims must be specified for {} in properties".format(
                    name
                )
            )
        elif "dims" not in input_properties[name].keys():
            raise InvalidPropertyDictError(
                "Input dims must be specified for {} in properties".format(
                    name
                )
            )
        else:
            return_array[name] = input_properties[name]["dims"]
    return return_array
