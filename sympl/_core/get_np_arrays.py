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

from .exceptions import InvalidStateError
from .wildcard import (
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


def get_numpy_array(data_array, out_dims, dim_lengths):
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
