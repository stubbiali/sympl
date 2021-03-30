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
    ConstantDiagnosticComponent,
    ConstantTendencyComponent,
    DataArray,
    RelaxationTendencyComponent,
    timedelta,
)


def test_constant_prognostic_empty_dicts():
    prog = ConstantTendencyComponent({}, {})
    tendencies, diagnostics = prog({"time": timedelta(0)})
    assert isinstance(tendencies, dict)
    assert isinstance(diagnostics, dict)
    assert len(tendencies) == 0
    assert len(diagnostics) == 0


def test_constant_prognostic_cannot_modify_through_input_dict():
    in_tendencies = {}
    in_diagnostics = {}
    prog = ConstantTendencyComponent(in_tendencies, in_diagnostics)
    in_tendencies["a"] = "b"
    in_diagnostics["c"] = "d"
    tendencies, diagnostics = prog({"time": timedelta(0)})
    assert len(tendencies) == 0
    assert len(diagnostics) == 0


def test_constant_prognostic_cannot_modify_through_output_dict():
    prog = ConstantTendencyComponent({}, {})
    tendencies, diagnostics = prog({"time": timedelta(0)})
    tendencies["a"] = "b"
    diagnostics["c"] = "d"
    tendencies, diagnostics = prog({"time": timedelta(0)})
    assert len(tendencies) == 0
    assert len(diagnostics) == 0


def test_constant_prognostic_tendency_properties():
    tendencies = {
        "tend1": DataArray(
            np.zeros([10]), dims=["dim1"], attrs={"units": "m/s"},
        ),
        "tend2": DataArray(
            np.zeros([2, 2]), dims=["dim2", "dim3"], attrs={"units": "degK/s"},
        ),
    }
    prog = ConstantTendencyComponent(tendencies)
    assert prog.tendency_properties == {
        "tend1": {"dims": ("dim1",), "units": "m/s",},
        "tend2": {"dims": ("dim2", "dim3"), "units": "degK/s"},
    }
    assert prog.diagnostic_properties == {}
    assert prog.input_properties == {}


def test_constant_prognostic_diagnostic_properties():
    tendencies = {}
    diagnostics = {
        "diag1": DataArray(
            np.zeros([10]), dims=["dim1"], attrs={"units": "m"},
        ),
        "diag2": DataArray(
            np.zeros([2, 2]), dims=["dim2", "dim3"], attrs={"units": "degK"},
        ),
    }
    prog = ConstantTendencyComponent(tendencies, diagnostics)
    assert prog.diagnostic_properties == {
        "diag1": {"dims": ("dim1",), "units": "m",},
        "diag2": {"dims": ("dim2", "dim3"), "units": "degK",},
    }
    assert prog.tendency_properties == {}
    assert prog.input_properties == {}


def test_constant_diagnostic_empty_dict():
    diag = ConstantDiagnosticComponent({})
    diagnostics = diag({"time": timedelta(0)})
    assert isinstance(diagnostics, dict)
    assert len(diagnostics) == 0


def test_constant_diagnostic_cannot_modify_through_input_dict():
    in_diagnostics = {}
    diag = ConstantDiagnosticComponent(in_diagnostics)
    in_diagnostics["a"] = "b"
    diagnostics = diag({"time": timedelta(0)})
    assert isinstance(diagnostics, dict)
    assert len(diagnostics) == 0


def test_constant_diagnostic_cannot_modify_through_output_dict():
    diag = ConstantDiagnosticComponent({})
    diagnostics = diag({"time": timedelta(0)})
    diagnostics["c"] = "d"
    diagnostics = diag({"time": timedelta(0)})
    assert len(diagnostics) == 0


def test_constant_diagnostic_diagnostic_properties():
    diagnostics = {
        "diag1": DataArray(
            np.zeros([10]), dims=["dim1"], attrs={"units": "m"},
        ),
        "diag2": DataArray(
            np.zeros([2, 2]), dims=["dim2", "dim3"], attrs={"units": "degK"},
        ),
    }
    diagnostic = ConstantDiagnosticComponent(diagnostics)
    assert diagnostic.diagnostic_properties == {
        "diag1": {"dims": ("dim1",), "units": "m",},
        "diag2": {"dims": ("dim2", "dim3"), "units": "degK",},
    }
    assert diagnostic.input_properties == {}


def test_relaxation_prognostic_at_equilibrium():
    prognostic = RelaxationTendencyComponent("quantity", "degK")
    state = {
        "time": timedelta(0),
        "quantity": DataArray(
            np.array([0.0, 1.0, 2.0]), attrs={"units": "degK"}
        ),
        "quantity_relaxation_timescale": DataArray(
            np.array([1.0, 1.0, 1.0]), attrs={"units": "s"}
        ),
        "equilibrium_quantity": DataArray(
            np.array([0.0, 1.0, 2.0]), attrs={"units": "degK"}
        ),
    }
    tendencies, diagnostics = prognostic(state)
    assert np.all(tendencies["quantity"].values == 0.0)


def test_relaxation_prognostic_with_change():
    prognostic = RelaxationTendencyComponent("quantity", "degK")
    state = {
        "time": timedelta(0),
        "quantity": DataArray(
            np.array([0.0, 1.0, 2.0]), attrs={"units": "degK"}
        ),
        "quantity_relaxation_timescale": DataArray(
            np.array([1.0, 1.0, 1.0]), attrs={"units": "s"}
        ),
        "equilibrium_quantity": DataArray(
            np.array([1.0, 3.0, 5.0]), attrs={"units": "degK"}
        ),
    }
    tendencies, diagnostics = prognostic(state)
    assert np.all(tendencies["quantity"].values == np.array([1.0, 2.0, 3.0]))


def test_relaxation_prognostic_with_change_different_timescale_units():
    prognostic = RelaxationTendencyComponent("quantity", "degK")
    state = {
        "time": timedelta(0),
        "quantity": DataArray(
            np.array([0.0, 1.0, 2.0]), attrs={"units": "degK"}
        ),
        "quantity_relaxation_timescale": DataArray(
            np.array([1 / 60.0, 2 / 60.0, 3 / 60.0]),
            attrs={"units": "minutes"},
        ),
        "equilibrium_quantity": DataArray(
            np.array([1.0, 3.0, 5.0]), attrs={"units": "degK"}
        ),
    }
    tendencies, diagnostics = prognostic(state)
    assert np.all(tendencies["quantity"].values == np.array([1.0, 1.0, 1.0]))


def test_relaxation_prognostic_with_change_different_equilibrium_units():
    prognostic = RelaxationTendencyComponent("quantity", "m")
    state = {
        "time": timedelta(0),
        "quantity": DataArray(np.array([0.0, 1.0, 2.0]), attrs={"units": "m"}),
        "quantity_relaxation_timescale": DataArray(
            np.array([1.0, 2.0, 3.0]), attrs={"units": "s"}
        ),
        "equilibrium_quantity": DataArray(
            np.array([1.0, 3.0, 5.0]) * 1e-3, attrs={"units": "km"}
        ),
    }
    tendencies, diagnostics = prognostic(state)
    assert np.all(tendencies["quantity"].values == np.array([1.0, 1.0, 1.0]))


if __name__ == "__main__":
    pytest.main([__file__])
