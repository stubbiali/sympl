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
from copy import deepcopy
import numpy as np
from typing import Any, Dict, Sequence, TYPE_CHECKING, Tuple

# from sympl._core.tracers import get_tracer_names
from sympl._core.checks import (
    check_array_shape,
    ensure_quantity_has_units,
    ensure_values_are_arrays,
)
from sympl._core.data_array import DataArray
from sympl._core.dims import (
    extract_output_dims_properties,
    fill_dims_wildcard,
    get_alias_or_name,
    get_wildcard_names_and_dim_lengths,
)
from sympl._core.exceptions import (
    InvalidPropertyDictError,
    InvalidStateError,
)

if TYPE_CHECKING:
    from sympl._core.typingx import (
        DataArrayDict,
        NDArrayLikeDict,
        PropertyDict,
    )


def get_arrays_with_properties(
    state: "DataArrayDict",
    property_dictionary: "PropertyDict",
    *,
    enable_checks: bool = True
) -> "NDArrayLikeDict":
    out_dict = {}
    wildcard_names, dim_lengths = get_wildcard_names_and_dim_lengths(
        state, property_dictionary, enable_checks=enable_checks
    )

    for name, properties in property_dictionary.items():
        if enable_checks:
            ensure_quantity_has_units(state[name], name)

        try:
            quantity = state[name].to_units(properties["units"])
        except ValueError:
            raise InvalidStateError(
                f"Could not convert quantity {name} from units "
                f"{state[name].attrs['units']} to units {properties['units']}."
            )

        out_dims = deepcopy(properties["dims"])
        has_wildcard = "*" in out_dims
        if has_wildcard:
            i_wildcard = out_dims.index("*")
            out_dims[i_wildcard : i_wildcard + 1] = wildcard_names

        out_array = get_array(
            quantity, out_dims=out_dims, dim_lengths=dim_lengths
        )
        if has_wildcard:
            out_array = flatten_wildcard_dims(
                out_array,
                i_wildcard,
                i_wildcard + len(wildcard_names),
                enable_checks=enable_checks,
            )

        if "alias" in properties:
            out_name = properties["alias"]
        else:
            out_name = name

        out_dict[out_name] = out_array

    return out_dict


def get_array(
    data_array: "DataArray",
    out_dims: Sequence[str],
    dim_lengths: Dict[str, int],
) -> "NDArrayLike":
    """
    Gets a numpy array from the data_array with the desired out_dims, and a
    dict of dim_lengths that will give the length of any missing dims in the
    data_array.
    """
    if len(data_array.shape) == 0 and len(out_dims) == 0:
        # special case, 0-dimensional scalar array
        return data_array.data
    else:
        missing_dims = [dim for dim in out_dims if dim not in data_array.dims]
        for dim in missing_dims:
            data_array = data_array.expand_dims(dim)

        if not all(
            dim1 == dim2 for dim1, dim2 in zip(data_array.dims, out_dims)
        ):
            array = data_array.transpose(*out_dims).data
        else:
            array = data_array.data

        if len(missing_dims) == 0:
            out_array = array
        else:
            # expand out missing dims which are currently length 1.
            out_shape = [dim_lengths.get(name, 1) for name in out_dims]
            if out_shape == list(array.shape):
                out_array = array
            else:
                # TODO: remove concrete dependency on numpy
                out_array = np.empty(out_shape, dtype=array.dtype)
                out_array[:] = array

        return out_array


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


def restore_dimensions(
    array: "NDArrayLike",
    from_dims: Sequence[str],
    result_like: "DataArray",
    result_attrs: Dict[str, Any] = None,
    *,
    enable_checks: bool = True
) -> "DataArray":
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
    # current_dim_names = {}
    # for dim in from_dims:
    #     if dim != "*":
    #         current_dim_names[dim] = dim
    # direction_to_names = get_array_dim_names(
    #     result_like, from_dims, current_dim_names, enable_checks=enable_checks
    # )
    #
    # original_shape = []
    # original_dims = []
    # original_coords = []
    # for direction in from_dims:
    #     if direction in direction_to_names:
    #         for name in direction_to_names[direction]:
    #             original_shape.append(len(result_like.coords[name]))
    #             original_dims.append(name)
    #             original_coords.append(result_like.coords[name])
    #
    # if enable_checks:
    #     if np.product(array.shape) != np.product(original_shape):
    #         raise ShapeMismatchError
    #
    # data_array = DataArray(
    #     np.reshape(array, original_shape),
    #     dims=original_dims,
    #     coords=original_coords,
    # ).transpose(*list(result_like.dims))
    #
    # if result_attrs is not None:
    #     data_array.attrs = result_attrs
    #
    # return data_array
    pass


def get_numpy_array(
    data_array,
    out_dims,
    return_wildcard_matches=False,
    require_wildcard_matches=None,
):
    """
    Retrieve a numpy array with the desired dimensions and dimension order
    from the given DataArray, using transpose and creating length 1 dimensions
    as necessary.

    Args
    ----
    data_array : DataArray
        The object from which to retrieve data.
    out_dims : list of str
        The desired dimensions of the output and their order.
        Length 1 dimensions will be created if the dimension is 'x', 'y', 'z',
        or '*' and does not exist in data_array. 'x', 'y', and 'z' indicate any axes
        registered to those directions with
        :py:function:`~sympl.set_direction_names`. '*' indicates an axis
        which is the flattened collection of all dimensions not explicitly
        listed in out_dims, including any dimensions with unknown direction.
    return_wildcard_matches : bool, optional
        If True, will additionally return a dictionary whose keys are direction
        wildcards (currently only '*') and values are lists of matched
        dimensions in the order they appear.
    require_wildcard_matches : dict, optional
        A dictionary mapping wildcards to matches. If the wildcard is used in
        out_dims, ensures that it matches the quantities present in this
        dictionary, in the same order.

    Returns
    -------
    numpy_array : ndarray
        The desired array, with dimensions in the
        correct order and length 1 dimensions created as needed.

    Raises
    ------
    ValueError
        If out_dims has values that are incompatible with the dimensions
        in data_array, or data_array's dimensions are invalid in some way.

    """
    # # This function was written when we had directional wildcards, and could
    # # be re-written to be simpler now that we do not.
    # if (len(data_array.values.shape) == 0) and (len(out_dims) == 0):
    #     direction_to_names = {}  # required in case we need wildcard_matches
    #     return_array = (
    #         data_array.values
    #     )  # special case, 0-dimensional scalar array
    # else:
    #     current_dim_names = {}
    #     for dim in out_dims:
    #         if dim != "*":
    #             current_dim_names[dim] = [dim]
    #     direction_to_names = get_array_dim_names(
    #         data_array, out_dims, current_dim_names
    #     )
    #     if require_wildcard_matches is not None:
    #         for direction in out_dims:
    #             if direction in require_wildcard_matches and same_list(
    #                 direction_to_names[direction],
    #                 require_wildcard_matches[direction],
    #             ):
    #                 direction_to_names[direction] = require_wildcard_matches[
    #                     direction
    #                 ]
    #             else:
    #                 # we could raise an exception here, because this is
    #                 # inconsistent, but that exception is already raised
    #                 # elsewhere when ensure_dims_like_are_satisfied is called
    #                 pass
    #     target_dimension_order = get_target_dimension_order(
    #         out_dims, direction_to_names
    #     )
    #     for dim in data_array.dims:
    #         if dim not in target_dimension_order:
    #             raise DimensionNotInOutDimsError(dim)
    #     slices_or_none = get_slices_and_placeholder_nones(
    #         data_array, out_dims, direction_to_names
    #     )
    #     final_shape = get_final_shape(data_array, out_dims, direction_to_names)
    #     return_array = np.reshape(
    #         data_array.transpose(*target_dimension_order).values[
    #             slices_or_none
    #         ],
    #         final_shape,
    #     )
    # if return_wildcard_matches:
    #     wildcard_matches = {
    #         key: value
    #         for key, value in direction_to_names.items()
    #         if key == "*"
    #     }
    #     return return_array, wildcard_matches
    # else:
    #     return return_array
    pass


def flatten_wildcard_dims(
    array: "NDArrayLike",
    i_start: int,
    i_end: int,
    *,
    enable_checks: bool = True
) -> "NDArrayLike":
    if enable_checks:
        if i_end > len(array.shape):
            raise ValueError(
                "i_end should be less than the number of axes in array"
            )
        elif i_start < 0:
            raise ValueError("i_start should be greater than 0")
        elif i_start > i_end:
            raise ValueError("i_start should be less than or equal to i_end")

    if i_start == i_end:
        # We need to insert a singleton dimension at i_start
        target_shape = deepcopy(array.shape)
        target_shape.insert(i_start, 1)
    else:
        target_shape = []
        wildcard_length = 1
        for i, length in enumerate(array.shape):
            if i_start <= i < i_end:
                wildcard_length *= length
            else:
                target_shape.append(length)
            if i == i_end - 1:
                target_shape.append(wildcard_length)

    return array.reshape(target_shape)


def expand_wildcard_dims(
    array: "NDArrayLike",
    target_shape: Sequence[int],
    name: str,
    out_dims: Sequence[str],
) -> "NDArrayLike":
    try:
        out_array = array.reshape(target_shape)
    except ValueError:
        raise InvalidPropertyDictError(
            f"Failed to restore shape for output {name} with original shape "
            f"{array.shape} and target shape {target_shape}, are the output "
            f"dims {out_dims} correct?"
        )
    return out_array


def restore_data_arrays_with_properties(
    raw_arrays: "NDArrayLikeDict",
    output_properties: "PropertyDict",
    input_state: "DataArrayDict",
    input_properties: "PropertyDict",
    ignore_names: Sequence[str] = None,
    ignore_missing: Sequence[str] = False,
    *,
    enable_checks: bool = True
) -> "DataArrayDict":
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
    # raw_arrays = raw_arrays.copy()
    if enable_checks:
        ensure_values_are_arrays(raw_arrays)

    ignore_names = ignore_names or []
    if ignore_missing:
        ignore_names = (
            set(output_properties.keys())
            .difference(raw_arrays.keys())
            .union(ignore_names)
        )

    wildcard_names, dim_lengths = get_wildcard_names_and_dim_lengths(
        input_state, input_properties, enable_checks=enable_checks
    )
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
            out_array = expand_wildcard_dims(
                raw_arrays[raw_name], target_shape, name, out_dims
            )
        else:
            if enable_checks:
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
