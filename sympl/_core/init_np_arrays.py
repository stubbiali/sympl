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
from .restore_dataarray import extract_output_dims_properties
from .tracers import get_tracer_names


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
    dim_lengths = get_dim_lengths_from_raw_input(
        raw_input_state, input_properties
    )
    dims_from_out_properties = extract_output_dims_properties(
        output_properties, input_properties, []
    )
    out_dict = {}
    tracer_names = list(get_tracer_names())
    tracer_names.extend(entry[0] for entry in prepend_tracers)
    for name, out_dims in dims_from_out_properties.items():
        if tracer_dims is None or name not in tracer_names:
            out_shape = []
            for dim in out_dims:
                out_shape.append(dim_lengths[dim])
            dtype = output_properties[name].get("dtype", np.float64)
            out_dict[name] = np.zeros(out_shape, dtype=dtype)
    if tracer_dims is not None:
        out_shape = []
        dim_lengths["tracer"] = len(tracer_names)
        for dim in tracer_dims:
            out_shape.append(dim_lengths[dim])
        out_dict["tracers"] = np.zeros(out_shape, dtype=np.float64)
    return out_dict


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
