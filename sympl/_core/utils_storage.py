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

# from sympl._core.tracers import get_tracer_names
from sympl._core.exceptions import InvalidStateError, ShapeMismatchError
from sympl._core.restore_dataarray import extract_output_dims_properties
from sympl._core.utils import get_input_array_dim_names
from sympl._core.wildcard import (
    flatten_wildcard_dims,
    get_wildcard_matches_and_dim_lengths,
)


def get_numpy_arrays_with_properties(state, property_dictionary):
    out_dict = {}
    wildcard_names, dim_lengths = get_wildcard_matches_and_dim_lengths(
        state, property_dictionary
    )
    #  Now we actually retrieve output arrays since we know the precise out dims
    for name, properties in property_dictionary.items():
        ensure_quantity_has_units(state[name], name)
        try:
            quantity = state[name].to_units(properties["units"])
        except ValueError:
            raise InvalidStateError(
                "Could not convert quantity {} from units {} to units {}".format(
                    name, state[name].attrs["units"], properties["units"]
                )
            )
        out_dims = []
        out_dims.extend(properties["dims"])
        has_wildcard = "*" in out_dims
        if has_wildcard:
            i_wildcard = out_dims.index("*")
            out_dims[i_wildcard : i_wildcard + 1] = wildcard_names
        out_array = get_numpy_array(
            quantity, out_dims=out_dims, dim_lengths=dim_lengths
        )
        if has_wildcard:
            out_array = flatten_wildcard_dims(
                out_array, i_wildcard, i_wildcard + len(wildcard_names)
            )
        if "alias" in properties.keys():
            out_name = properties["alias"]
        else:
            out_name = name
        out_dict[out_name] = out_array
    return out_dict


def get_array(data_array, out_dims, dim_lengths):
    """
    Gets a numpy array from the data_array with the desired out_dims, and a
    dict of dim_lengths that will give the length of any missing dims in the
    data_array.
    """
    if len(data_array.data.shape) == 0 and len(out_dims) == 0:
        return data_array.data  # special case, 0-dimensional scalar array
    else:
        missing_dims = [dim for dim in out_dims if dim not in data_array.dims]
        for dim in missing_dims:
            data_array = data_array.expand_dims(dim)
        if not all(
            dim1 == dim2 for dim1, dim2 in zip(data_array.dims, out_dims)
        ):
            numpy_array = data_array.transpose(*out_dims).data
        else:
            numpy_array = data_array.data
        if len(missing_dims) == 0:
            out_array = numpy_array
        else:  # expand out missing dims which are currently length 1.
            out_shape = [dim_lengths.get(name, 1) for name in out_dims]
            if out_shape == list(numpy_array.shape):
                out_array = numpy_array
            else:
                out_array = np.empty(out_shape, dtype=numpy_array.dtype)
                out_array[:] = numpy_array
        return out_array


def ensure_quantity_has_units(quantity, quantity_name):
    if "units" not in quantity.attrs:
        raise InvalidStateError(
            "quantity {} is missing units attribute".format(quantity_name)
        )


def initialize_numpy_arrays_with_properties(
    output_properties,
    raw_input_state,
    input_properties,
    tracer_dims=None,
    prepend_tracers=(),
):
    """
    Parameters
    ----------
    output_properties : dict
        A dictionary whose keys are quantity names and values are dictionaries
        with properties for those quantities. The property "dims" must be
        present for each quantity not also present in input_properties.
    raw_input_state : dict
        A state dictionary of numpy arrays that was used as input to a component
        for which return arrays are being generated.
    input_properties : dict
        A dictionary whose keys are quantity names and values are dictionaries
        with input properties for those quantities. The property "dims" must be
        present, indicating the dimensions that the quantity was transformed to
        when taken as input to a component.

    Returns
    -------
    out_dict : dict
        A dictionary whose keys are quantities and values are numpy arrays
        corresponding to those quantities, with shapes determined from the
        inputs to this function.

    Raises
    ------
    InvalidPropertyDictError
        When an output property is specified to have dims_like an input
        property, but the arrays for the two properties have incompatible
        shapes.
    """
    # dim_lengths = get_dim_lengths_from_raw_input(
    #     raw_input_state, input_properties
    # )
    # dims_from_out_properties = extract_output_dims_properties(
    #     output_properties, input_properties, []
    # )
    # out_dict = {}
    # tracer_names = list(get_tracer_names())
    # tracer_names.extend(entry[0] for entry in prepend_tracers)
    # for name, out_dims in dims_from_out_properties.items():
    #     if tracer_dims is None or name not in tracer_names:
    #         out_shape = []
    #         for dim in out_dims:
    #             out_shape.append(dim_lengths[dim])
    #         dtype = output_properties[name].get("dtype", np.float64)
    #         out_dict[name] = np.zeros(out_shape, dtype=dtype)
    # if tracer_dims is not None:
    #     out_shape = []
    #     dim_lengths["tracer"] = len(tracer_names)
    #     for dim in tracer_dims:
    #         out_shape.append(dim_lengths[dim])
    #     out_dict["tracers"] = np.zeros(out_shape, dtype=np.float64)
    # return out_dict
    pass


def get_dim_lengths_from_raw_input(raw_input, input_properties):
    dim_lengths = {}
    for name, properties in input_properties.items():
        if properties.get("tracer", False):
            continue
        if "alias" in properties.keys():
            name = properties["alias"]
        for i, dim_name in enumerate(properties["dims"]):
            if dim_name in dim_lengths:
                if raw_input[name].shape[i] != dim_lengths[dim_name]:
                    raise InvalidStateError(
                        "Dimension name {} has differing lengths on different "
                        "inputs".format(dim_name)
                    )
            else:
                dim_lengths[dim_name] = raw_input[name].shape[i]
    return dim_lengths


def restore_dimensions(array, from_dims, result_like, result_attrs=None):
    """
    Restores a numpy array to a DataArray with similar dimensions to a reference
    Data Array. This is meant to be the reverse of get_numpy_array.

    Parameters
    ----------
    array : ndarray
        The numpy array from which to create a DataArray
    from_dims : list of str
        The directions describing the numpy array. If being used to reverse
        a call to get_numpy_array, this should be the same as the out_dims
        argument used in the call to get_numpy_array.
        'x', 'y', and 'z' indicate any axes
        registered to those directions with
        :py:function:`~sympl.set_direction_names`. '*' indicates an axis
        which is the flattened collection of all dimensions not explicitly
        listed in out_dims, including any dimensions with unknown direction.
    result_like : DataArray
        A reference array with the desired output dimensions of the DataArray.
        If being used to reverse a call to get_numpy_array, this should be
        the same as the data_array argument used in the call to get_numpy_array.
    result_attrs : dict, optional
        A dictionary with the desired attributes of the output DataArray. If
        not given, no attributes will be set.

    Returns
    -------
    data_array : DataArray
        The output DataArray with the same dimensions as the reference
        DataArray.

    See Also
    --------
    :py:function:~sympl.get_numpy_array: : Retrieves a numpy array with desired
        dimensions from a given DataArray.
    """
    current_dim_names = {}
    for dim in from_dims:
        if dim != "*":
            current_dim_names[dim] = [dim]
    direction_to_names = get_input_array_dim_names(
        result_like, from_dims, current_dim_names
    )
    original_shape = []
    original_dims = []
    original_coords = []
    for direction in from_dims:
        if direction in direction_to_names.keys():
            for name in direction_to_names[direction]:
                original_shape.append(len(result_like.coords[name]))
                original_dims.append(name)
                original_coords.append(result_like.coords[name])
    if np.product(array.shape) != np.product(original_shape):
        raise ShapeMismatchError
    data_array = DataArray(
        np.reshape(array, original_shape),
        dims=original_dims,
        coords=original_coords,
    ).transpose(*list(result_like.dims))
    if result_attrs is not None:
        data_array.attrs = result_attrs
    return data_array
