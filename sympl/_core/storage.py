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
from typing import Sequence, TYPE_CHECKING

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
    pass


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
    pass
