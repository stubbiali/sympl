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
import pytest

from sympl import (
    DataArray,
    InvalidStateError,
    get_numpy_arrays_with_properties,
)


def test_match_dims_like_hardcoded_dimensions_matching_lengths():
    input_state = {
        "air_temperature": DataArray(
            np.zeros([2, 3, 4]),
            dims=["alpha", "beta", "gamma"],
            attrs={"units": "degK"},
        ),
        "air_pressure": DataArray(
            np.zeros([2, 3, 4]),
            dims=["alpha", "beta", "gamma"],
            attrs={"units": "Pa"},
        ),
    }
    input_properties = {
        "air_temperature": {
            "dims": ["alpha", "beta", "gamma"],
            "units": "degK",
            "match_dims_like": "air_pressure",
        },
        "air_pressure": {"dims": ["alpha", "beta", "gamma"], "units": "Pa",},
    }
    raw_arrays = get_numpy_arrays_with_properties(
        input_state, input_properties
    )


def test_match_dims_like_partly_hardcoded_dimensions_matching_lengths():
    input_state = {
        "air_temperature": DataArray(
            np.zeros([2, 3, 4]),
            dims=["lat", "lon", "mid_levels"],
            attrs={"units": "degK"},
        ),
        "air_pressure": DataArray(
            np.zeros([2, 3, 4]),
            dims=["lat", "lon", "interface_levels"],
            attrs={"units": "Pa"},
        ),
    }
    input_properties = {
        "air_temperature": {
            "dims": ["*", "mid_levels"],
            "units": "degK",
            "match_dims_like": "air_pressure",
        },
        "air_pressure": {"dims": ["*", "interface_levels"], "units": "Pa",},
    }
    raw_arrays = get_numpy_arrays_with_properties(
        input_state, input_properties
    )
    assert np.byte_bounds(
        input_state["air_temperature"].values
    ) == np.byte_bounds(raw_arrays["air_temperature"])
    assert np.byte_bounds(
        input_state["air_pressure"].values
    ) == np.byte_bounds(raw_arrays["air_pressure"])


def test_match_dims_like_hardcoded_dimensions_non_matching_lengths():
    input_state = {
        "air_temperature": DataArray(
            np.zeros([2, 3, 4]),
            dims=["alpha", "beta", "gamma"],
            attrs={"units": "degK"},
        ),
        "air_pressure": DataArray(
            np.zeros([4, 2, 3]),
            dims=["alpha", "beta", "gamma"],
            attrs={"units": "Pa"},
        ),
    }
    input_properties = {
        "air_temperature": {
            "dims": ["alpha", "beta", "gamma"],
            "units": "degK",
            "match_dims_like": "air_pressure",
        },
        "air_pressure": {"dims": ["alpha", "beta", "gamma"], "units": "Pa",},
    }
    try:
        raw_arrays = get_numpy_arrays_with_properties(
            input_state, input_properties
        )
    except InvalidStateError:
        pass
    else:
        raise AssertionError("should have raised InvalidStateError")


def test_match_dims_like_wildcard_dimensions_matching_lengths():
    input_state = {
        "air_temperature": DataArray(
            np.zeros([2, 3, 4]),
            dims=["alpha", "beta", "gamma"],
            attrs={"units": "degK"},
        ),
        "air_pressure": DataArray(
            np.zeros([2, 3, 4]),
            dims=["alpha", "beta", "gamma"],
            attrs={"units": "Pa"},
        ),
    }
    input_properties = {
        "air_temperature": {
            "dims": ["*"],
            "units": "degK",
            "match_dims_like": "air_pressure",
        },
        "air_pressure": {"dims": ["*"], "units": "Pa",},
    }
    raw_arrays = get_numpy_arrays_with_properties(
        input_state, input_properties
    )


def test_match_dims_like_wildcard_dimensions_non_matching_lengths():
    input_state = {
        "air_temperature": DataArray(
            np.zeros([2, 3, 4]),
            dims=["alpha", "beta", "gamma"],
            attrs={"units": "degK"},
        ),
        "air_pressure": DataArray(
            np.zeros([1, 2, 3]),
            dims=["alpha", "beta", "gamma"],
            attrs={"units": "Pa"},
        ),
    }
    input_properties = {
        "air_temperature": {
            "dims": ["*"],
            "units": "degK",
            "match_dims_like": "air_pressure",
        },
        "air_pressure": {"dims": ["*"], "units": "Pa",},
    }
    try:
        raw_arrays = get_numpy_arrays_with_properties(
            input_state, input_properties
        )
    except InvalidStateError:
        pass
    else:
        raise AssertionError("should have raised InvalidStateError")


def test_match_dims_like_wildcard_dimensions_use_same_ordering():
    input_state = {
        "air_temperature": DataArray(
            np.random.randn(2, 3, 4),
            dims=["alpha", "beta", "gamma"],
            attrs={"units": "degK"},
        ),
        "air_pressure": DataArray(
            np.zeros([4, 2, 3]),
            dims=["gamma", "alpha", "beta"],
            attrs={"units": "Pa"},
        ),
    }
    for i in range(4):
        input_state["air_pressure"][i, :, :] = input_state["air_temperature"][
            :, :, i
        ]
    input_properties = {
        "air_temperature": {
            "dims": ["*"],
            "units": "degK",
            "match_dims_like": "air_pressure",
        },
        "air_pressure": {"dims": ["*"], "units": "Pa",},
    }
    raw_arrays = get_numpy_arrays_with_properties(
        input_state, input_properties
    )
    assert np.all(raw_arrays["air_temperature"] == raw_arrays["air_pressure"])


if __name__ == "__main__":
    pytest.main([__file__])
