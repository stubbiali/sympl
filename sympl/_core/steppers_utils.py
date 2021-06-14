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
from typing import Optional, TYPE_CHECKING, Tuple

from sympl._core.dynamic_operators import OutflowComponentOperator
from sympl._core.static_operators import StaticComponentOperator
from sympl._core.units import clean_units

if TYPE_CHECKING:
    from datetime import timedelta

    from sympl._core.steppers import TendencyStepper
    from sympl._core.typingx import DataArray, DataArrayDict, PropertyDict


class StaticOperator:
    input_operator = StaticComponentOperator.factory("input_properties")
    tendency_operator = StaticComponentOperator.factory("tendency_properties")
    diagnostic_operator = StaticComponentOperator.factory(
        "diagnostic_properties"
    )

    @classmethod
    def get_input_properties(
        cls, stepper: "TendencyStepper"
    ) -> "PropertyDict":
        out = {}
        out.update(
            cls.input_operator.get_properties_with_dims(
                stepper.tendency_component
            )
        )
        tc_tendency_properties = cls.tendency_operator.get_properties_with_dims(
            stepper.tendency_component
        )
        for name in tc_tendency_properties:
            if name not in out:
                out[name] = tc_tendency_properties[name].copy()
                out[name]["units"] = clean_units(out[name]["units"] + " s")
        return out

    @classmethod
    def get_diagnostic_properties(
        cls, stepper: "TendencyStepper"
    ) -> "PropertyDict":
        return cls.diagnostic_operator.get_properties_with_dims(
            stepper.tendency_component
        )

    @classmethod
    def get_output_properties(
        cls, stepper: "TendencyStepper"
    ) -> "PropertyDict":
        out = {}
        tc_tendency_properties = cls.tendency_operator.get_properties_with_dims(
            stepper.tendency_component
        )
        for name in tc_tendency_properties:
            out[name] = tc_tendency_properties[name].copy()
            out[name]["units"] = clean_units(out[name]["units"] + " s")
        return out


class DynamicOperator:
    properties_name: str = None

    def __init__(self, stepper: "TendencyStepper") -> None:
        self.stepper = stepper
        self.sco_tendencies = StaticComponentOperator.factory(
            "tendency_properties"
        )
        self.sco_diagnostics = StaticComponentOperator.factory(
            "diagnostic_properties"
        )
        self.dco_tendencies = OutflowComponentOperator.factory(
            "output_properties", stepper
        )
        self.dco_diagnostics = OutflowComponentOperator.factory(
            "diagnostic_properties", stepper
        )
        self.dco_output = OutflowComponentOperator.factory(
            "output_properties", stepper
        )

    def allocate_diagnostic(
        self, name: str, state: "DataArrayDict"
    ) -> Optional["DataArray"]:
        allocator = self.sco_diagnostics.get_allocator(
            self.stepper.tendency_component
        )
        if allocator is not None:
            raw_diagnostic = allocator(name)
            diagnostic = self.dco_diagnostics.get_dataarray(
                name, raw_diagnostic, state
            )
            return diagnostic
        else:
            return None

    def allocate_diagnostic_dict(
        self, state: "DataArrayDict"
    ) -> "DataArrayDict":
        properties = self.sco_diagnostics.get_properties(
            self.stepper.tendency_component
        )
        allocator = self.sco_diagnostics.get_allocator(
            self.stepper.tendency_component
        )
        if allocator is not None:
            raw_diagnostics = {name: allocator(name) for name in properties}
        else:
            raw_diagnostics = {}
        out = self.dco_diagnostics.get_dataarray_dict(raw_diagnostics, state)
        return out

    def allocate_output(
        self, name: str, state: "DataArrayDict"
    ) -> Optional["DataArray"]:
        allocator = self.sco_tendencies.get_allocator(
            self.stepper.tendency_component
        )
        if allocator is not None:
            raw_out = allocator(name)
            out = self.dco_output.get_dataarray(name, raw_out, state)
            return out
        else:
            return None

    def allocate_output_dict(self, state: "DataArrayDict") -> "DataArrayDict":
        properties = self.sco_tendencies.get_properties(
            self.stepper.tendency_component
        )
        allocator = self.sco_tendencies.get_allocator(
            self.stepper.tendency_component
        )
        if allocator is not None:
            raw_out = {name: allocator(name) for name in properties}
        else:
            raw_out = {}
        out = self.dco_output.get_dataarray_dict(raw_out, state)
        return out

    def get_increment(
        self,
        state: "DataArrayDict",
        timestep: "timedelta",
        out_increment: Optional["DataArrayDict"],
        out_diagnostics: Optional["DataArrayDict"],
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        if out_increment is not None:
            for name in out_increment:
                if name != "time":
                    out_increment[name].attrs["units"] += " s^-1"

        try:
            out_increment, out_diagnostics = self.stepper.tendency_component(
                state,
                timestep,
                out_tendencies=out_increment,
                out_diagnostics=out_diagnostics,
            )
        except TypeError:
            out_increment, out_diagnostics = self.stepper.tendency_component(
                state,
                out_tendencies=out_increment,
                out_diagnostics=out_diagnostics,
            )

        for name in out_increment:
            if name != "time":
                out_increment[name].attrs["units"] += " s"

        return out_increment, out_diagnostics
