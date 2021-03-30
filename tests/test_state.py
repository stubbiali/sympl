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
import unittest

import numpy as np

from sympl import DataArray, initialize_numpy_arrays_with_properties


class InitializeNumpyArraysWithPropertiesTests(unittest.TestCase):
    def test_empty(self):
        output_properties = {}
        input_properties = {}
        input_state = {}
        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties
        )
        assert result == {}

    def test_single_output_single_dim(self):
        output_properties = {"output1": {"dims": ["dim1"], "units": "m",}}
        input_properties = {"input1": {"dims": ["dim1"], "units": "s^-1",}}
        input_state = {"input1": np.zeros([10])}

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties
        )
        assert len(result.keys()) == 1
        assert "output1" in result.keys()
        assert result["output1"].shape == (10,)
        assert np.all(result["output1"] == np.zeros([10]))
        assert result["output1"].dtype == np.float64

    def test_single_output_single_dim_aliased(self):
        output_properties = {"output1": {"dims": ["dim1"], "units": "m",}}
        input_properties = {
            "input1": {"dims": ["dim1"], "units": "s^-1", "alias": "in1"}
        }
        input_state = {"in1": np.zeros([10])}

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties
        )
        assert len(result.keys()) == 1
        assert "output1" in result.keys()
        assert result["output1"].shape == (10,)
        assert np.all(result["output1"] == np.zeros([10]))
        assert result["output1"].dtype == np.float64

    def test_single_output_single_dim_custom_dtype(self):
        output_properties = {
            "output1": {"dims": ["dim1"], "units": "m", "dtype": np.int32,}
        }
        input_properties = {"input1": {"dims": ["dim1"], "units": "s^-1",}}
        input_state = {"input1": np.zeros([10])}

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties
        )
        assert len(result.keys()) == 1
        assert "output1" in result.keys()
        assert result["output1"].shape == (10,)
        assert np.all(result["output1"] == np.zeros([10]))
        assert result["output1"].dtype == np.int32

    def test_single_output_two_dims(self):
        output_properties = {
            "output1": {"dims": ["dim1", "dim2"], "units": "m",}
        }
        input_properties = {
            "input1": {"dims": ["dim1", "dim2"], "units": "s^-1",}
        }
        input_state = {"input1": np.zeros([3, 7])}

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties
        )
        assert len(result.keys()) == 1
        assert "output1" in result.keys()
        assert result["output1"].shape == (3, 7)
        assert np.all(result["output1"] == np.zeros([3, 7]))

    def test_single_output_two_dims_opposite_order(self):
        output_properties = {
            "output1": {"dims": ["dim2", "dim1"], "units": "m",}
        }
        input_properties = {
            "input1": {"dims": ["dim1", "dim2"], "units": "s^-1",}
        }
        input_state = {"input1": np.zeros([3, 7])}

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties
        )
        assert len(result.keys()) == 1
        assert "output1" in result.keys()
        assert result["output1"].shape == (7, 3)
        assert np.all(result["output1"] == np.zeros([7, 3]))

    def test_two_outputs(self):
        output_properties = {
            "output1": {"dims": ["dim2", "dim1"], "units": "m",},
            "output2": {"dims": ["dim1", "dim2"], "units": "m",},
        }
        input_properties = {
            "input1": {"dims": ["dim1", "dim2"], "units": "s^-1",}
        }
        input_state = {"input1": np.zeros([3, 7])}

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties
        )
        assert len(result.keys()) == 2
        assert "output1" in result.keys()
        assert result["output1"].shape == (7, 3)
        assert np.all(result["output1"] == np.zeros([7, 3]))
        assert "output2" in result.keys()
        assert result["output2"].shape == (3, 7)
        assert np.all(result["output2"] == np.zeros([3, 7]))

    def test_two_inputs(self):
        output_properties = {
            "output1": {"dims": ["dim2", "dim1"], "units": "m",},
        }
        input_properties = {
            "input1": {"dims": ["dim1", "dim2"], "units": "s^-1",},
            "input2": {"dims": ["dim2", "dim1"], "units": "s^-1",},
        }
        input_state = {
            "input1": np.zeros([3, 7]),
            "input2": np.zeros([7, 3]),
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties
        )
        assert len(result.keys()) == 1
        assert "output1" in result.keys()
        assert result["output1"].shape == (7, 3)
        assert np.all(result["output1"] == np.zeros([7, 3]))

    def test_single_dim_wildcard(self):
        output_properties = {"output1": {"dims": ["*"], "units": "m",}}
        input_properties = {"input1": {"dims": ["*"], "units": "s^-1",}}
        input_state = {"input1": np.zeros([10])}

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties
        )
        assert len(result.keys()) == 1
        assert "output1" in result.keys()
        assert result["output1"].shape == (10,)
        assert np.all(result["output1"] == np.zeros([10]))
