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
import abc
from datetime import timedelta
from typing import List, Optional, TYPE_CHECKING, Tuple, Union

from sympl._core.base_component import BaseComponent
from sympl._core.core_components import ImplicitTendencyComponent
from sympl._core.dynamic_checkers import (
    InflowComponentChecker,
    OutflowComponentChecker,
)
from sympl._core.dynamic_operators import (
    InflowComponentOperator,
    OutflowComponentOperator,
)
from sympl._core.exceptions import InvalidPropertyDictError
from sympl._core.static_checkers import StaticComponentChecker
from sympl._core.steppers_utils import DynamicOperator, StaticOperator
from sympl._core.tracers import TracerPacker

if TYPE_CHECKING:
    from sympl._core.core_components import (
        ImplicitTendencyComponent,
        TendencyComponent,
    )
    from sympl._core.typingx import (
        DataArrayDict,
        NDArrayLikeDict,
        PropertyDict,
    )


class Stepper(BaseComponent):
    """TODO."""

    time_unit_name = "s"
    time_unit_timedelta = timedelta(seconds=1)
    uses_tracers = False
    tracer_dims = None

    def __init__(
        self,
        tendencies_in_diagnostics: bool = False,
        name: Optional[str] = None,
        *,
        enable_checks: bool = True
    ) -> None:
        """TODO."""
        super().__init__()

        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__
        self._enable_checks = enable_checks

        if enable_checks:
            StaticComponentChecker.factory("input_properties").check(self)
            StaticComponentChecker.factory("diagnostic_properties").check(self)
            StaticComponentChecker.factory("output_properties").check(self)

            self._input_checker = InflowComponentChecker.factory(
                "input_properties", self
            )
            self._diagnostic_inflow_checker = InflowComponentChecker.factory(
                "diagnostic_properties", self
            )
            self._diagnostic_outflow_checker = OutflowComponentChecker.factory(
                "diagnostic_properties", self
            )
            self._output_inflow_checker = InflowComponentChecker.factory(
                "output_properties", self
            )
            self._output_outflow_checker = OutflowComponentChecker.factory(
                "output_properties", self
            )

            if tendencies_in_diagnostics:
                self._diagnostic_outflow_checker.ignored_diagnostics = (
                    self._insert_tendency_properties()
                )

        if self.uses_tracers:
            if self.tracer_dims is None:
                raise ValueError(
                    f"Component of type {self.__class__.__name__} must specify "
                    f"tracer_dims property when uses_tracers=True."
                )
            prepend_tracers = getattr(self, "prepend_tracers", None)
            self._tracer_packer = TracerPacker(
                self, self.tracer_dims, prepend_tracers=prepend_tracers
            )

        self._input_operator = InflowComponentOperator.factory(
            "input_properties", self
        )
        self._diagnostic_inflow_operator = InflowComponentOperator.factory(
            "diagnostic_properties", self
        )
        self._diagnostic_outflow_operator = OutflowComponentOperator.factory(
            "diagnostic_properties", self
        )
        self._output_inflow_operator = InflowComponentOperator.factory(
            "output_properties", self
        )
        self._output_outflow_operator = OutflowComponentOperator.factory(
            "output_properties", self
        )

    def __str__(self) -> str:
        return (
            f"Instance of {self.__class__.__name__}(Stepper)\n"
            f"    inputs: {', '.join(self.input_properties.keys())}\n"
            f"    outputs: {', '.join(self.output_properties.keys())}\n"
            f"    diagnostics: {', '.join(self.diagnostic_properties.keys())}"
        )

    def __repr__(self) -> str:
        if hasattr(self, "_making_repr") and self._making_repr:
            return f"{self.__class__.__name__}(recursive reference)"
        else:
            self._making_repr = True
            return_value = "{}({})".format(
                self.__class__,
                "\n".join(
                    "{}: {}".format(repr(key), repr(value))
                    for key, value in self.__dict__.items()
                    if key != "_making_repr"
                ),
            )
            self._making_repr = False
            return return_value

    def __call__(
        self,
        state: "DataArrayDict",
        timestep: timedelta,
        *,
        out_diagnostics: Optional["DataArrayDict"],
        out_state: Optional["DataArrayDict"]
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        """
        Gets diagnostics from the current model state and steps the state
        forward in time according to the timestep.

        Args
        ----
        state : dict
            A model state dictionary satisfying the input_properties of this
            object.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state.
        new_state : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the timestep after input state.

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the Stepper instance
            for other reasons.
        """
        # inflow checks
        if self._enable_checks:
            self._input_checker.check(state)

        # extract raw state
        raw_state = self._input_operator.get_ndarray_dict(state)
        if self.uses_tracers:
            raw_state["tracers"] = self._tracer_packer.pack(state)
        raw_state["time"] = state["time"]

        if out_diagnostics is None:
            # allocate buffers for diagnostics
            raw_diagnostics = {
                name: self.allocate_diagnostic(name)
                for name in self.diagnostic_properties
            }

            # run checks on raw_diagnostics
            if self._enable_checks:
                self._diagnostic_outflow_checker.check(raw_diagnostics, state)
        else:
            # run checks on out_diagnostics
            if self._enable_checks:
                self._diagnostic_inflow_checker.check(out_diagnostics)

            # extract buffers for diagnostics
            raw_diagnostics = self._diagnostic_inflow_operator.get_ndarray_dict(
                out_diagnostics
            )

        if out_state is None:
            # allocate buffers for new state
            raw_new_state = {
                name: self.allocate_output(name)
                for name in self.output_properties
            }

            # run checks on raw_new_state
            if self._enable_checks:
                self._output_outflow_checker.check(raw_new_state, state)
        else:
            # run checks on out_state
            if self._enable_checks:
                self._output_inflow_checker.check(out_state)

            # extract buffers for diagnostics
            raw_new_state = self._output_inflow_operator.get_ndarray_dict(
                out_state
            )

        # compute
        raw_diagnostics, raw_new_state = self.array_call(
            raw_state, timestep, raw_diagnostics, raw_new_state
        )
        # if self.uses_tracers:
        #     new_state = self._tracer_packer.unpack(
        #         raw_new_state.pop("tracers"), state
        #     )
        # else:
        #     new_state = {}

        # outflow checks
        if self._enable_checks:
            self._diagnostic_outflow_checker.check(raw_diagnostics, state)
            self._output_outflow_checker.check_outputs(raw_new_state, state)

        # compute first-order approximation to tendencies
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(
                raw_state, raw_new_state, timestep, raw_diagnostics
            )

        # wrap output arrays in dataarrays
        diagnostics = self._diagnostic_outflow_operator.get_dataarray_dict(
            raw_diagnostics, state, out=out_diagnostics
        )
        new_state = self._output_outflow_operator.get_dataarray_dict(
            raw_new_state, state, out=out_state
        )

        return diagnostics, new_state

    @property
    def tendencies_in_diagnostics(self) -> bool:
        return self._tendencies_in_diagnostics  # value cannot be modified

    @property
    @abc.abstractmethod
    def input_properties(self) -> "PropertyDict":
        pass

    @property
    @abc.abstractmethod
    def diagnostic_properties(self) -> "PropertyDict":
        pass

    @property
    @abc.abstractmethod
    def output_properties(self) -> "PropertyDict":
        pass

    @abc.abstractmethod
    def allocate_diagnostic(self, name) -> "NDArrayLikeDict":
        pass

    @abc.abstractmethod
    def allocate_output(self, name) -> "NDArrayLikeDict":
        pass

    @abc.abstractmethod
    def array_call(
        self,
        state: "NDArrayLikeDict",
        timestep: timedelta,
        out_diagnostics: "NDArrayLikeDict",
        out_state: "NDArrayLikeDict",
    ) -> Tuple["NDArrayLikeDict", "NDArrayLikeDict"]:
        """TODO."""

    def _insert_tendency_properties(self) -> List[str]:
        added_names = []
        for name, properties in self.output_properties.items():
            tendency_name = self._get_tendency_name(name)
            if properties["units"] == "":
                units = "s^-1"
            else:
                units = "{} s^-1".format(properties["units"])
            if "dims" in properties.keys():
                dims = properties["dims"]
            else:
                dims = self.input_properties[name]["dims"]
            self.diagnostic_properties[tendency_name] = {
                "units": units,
                "dims": dims,
            }
            if name not in self.input_properties.keys():
                self.input_properties[name] = {
                    "dims": dims,
                    "units": properties["units"],
                }
            elif self.input_properties[name]["dims"] != dims:
                raise InvalidPropertyDictError(
                    "Can only calculate tendencies when input and output values"
                    " have the same dimensions, but dims for {} are "
                    "{} (input) and {} (output)".format(
                        name,
                        self.input_properties[name]["dims"],
                        self.output_properties[name]["dims"],
                    )
                )
            elif (
                self.input_properties[name]["units"]
                != self.output_properties[name]["units"]
            ):
                raise InvalidPropertyDictError(
                    "Can only calculate tendencies when input and output values"
                    " have the same units, but units for {} are "
                    "{} (input) and {} (output)".format(
                        name,
                        self.input_properties[name]["units"],
                        self.output_properties[name]["units"],
                    )
                )
            added_names.append(tendency_name)
        return added_names

    def _get_tendency_name(self, name: str) -> str:
        return f"{name}_tendency_from_{self.name}"

    def _insert_tendencies_to_diagnostics(
        self,
        raw_state: "NDArrayLikeDict",
        raw_new_state: "NDArrayLikeDict",
        timestep: timedelta,
        raw_diagnostics: "NDArrayLikeDict",
    ) -> None:
        for name in self.output_properties.keys():
            tendency_name = self._get_tendency_name(name)
            raw_diagnostics[tendency_name] = (
                (raw_new_state[name] - raw_state[name])
                / timestep.total_seconds()
                * self.time_unit_timedelta.total_seconds()
            )


class TendencyStepper(BaseComponent):
    """TODO."""

    time_unit_name = "s"
    time_unit_timedelta = timedelta(seconds=1)

    def __init__(
        self,
        tendency_component: Union[
            "ImplicitTendencyComponent", "TendencyComponent"
        ],
        tendencies_in_diagnostics: bool = False,
        name: Optional[str] = None,
        *,
        enable_checks: bool = True
    ) -> None:
        """TODO."""
        super().__init__()

        self.tendency_component = tendency_component
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__
        self._enable_checks = enable_checks

        self.input_properties = StaticOperator.get_input_properties(self)
        self.diagnostic_properties = StaticOperator.get_diagnostic_properties(
            self
        )
        self.output_properties = StaticOperator.get_output_properties(self)

        if enable_checks:
            StaticComponentChecker.factory("input_properties").check(self)
            StaticComponentChecker.factory("diagnostic_properties").check(self)
            StaticComponentChecker.factory("output_properties").check(self)

            self._input_checker = InflowComponentChecker.factory(
                "input_properties", self
            )
            self._diagnostic_checker = InflowComponentChecker.factory(
                "diagnostic_properties", self
            )
            self._output_checker = InflowComponentChecker.factory(
                "output_properties", self
            )

        self._stepper_operator = DynamicOperator(self)

    def __str__(self) -> str:
        return (
            f"Instance of {self.__class__.__name__}(TendencyStepper)\n"
            f"    inputs: {', '.join(self.input_properties.keys())}\n"
            f"    outputs: {', '.join(self.output_properties.keys())}\n"
            f"    diagnostics: {', '.join(self.diagnostic_properties.keys())}"
        )

    def __repr__(self) -> str:
        if hasattr(self, "_making_repr") and self._making_repr:
            return f"{self.__class__.__name__}(recursive reference)"
        else:
            self._making_repr = True
            return_value = "{}({})".format(
                self.__class__,
                "\n".join(
                    "{}: {}".format(repr(key), repr(value))
                    for key, value in self.__dict__.items()
                    if key != "_making_repr"
                ),
            )
            self._making_repr = False
            return return_value

    def __call__(
        self,
        state: "DataArrayDict",
        timestep: timedelta,
        *,
        out_diagnostics: Optional["DataArrayDict"] = None,
        out_state: Optional["DataArrayDict"] = None
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        """
        Gets diagnostics from the current model state and steps the state
        forward in time according to the timestep.

        Args
        ----
        state : dict
            A model state dictionary satisfying the input_properties of this
            object.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state.
        new_state : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the timestep after input state.

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the Stepper instance
            for other reasons.
        """
        if self._enable_checks:
            self._input_checker.check(state)

        out_diagnostics = (
            out_diagnostics if out_diagnostics is not None else {}
        )
        out_state = out_state if out_state is not None else {}

        if self._enable_checks:
            self._diagnostic_checker.check(out_diagnostics, state)
            self._output_checker.check(out_state, state)

        out_diagnostics.update(
            {
                name: self._stepper_operator.allocate_diagnostic(name, state)
                for name in self.diagnostic_properties
                if name not in out_diagnostics
            }
        )
        out_state.update(
            {
                name: self._stepper_operator.allocate_output(name, state)
                for name in self.output_properties
                if name not in out_state
            }
        )

        self._call(state, timestep, out_diagnostics, out_state)

        if self._enable_checks:
            self._diagnostic_checker.check(out_diagnostics, state)
            self._output_checker.check(out_state, state)

        return out_diagnostics, out_state

    @property
    def tendencies_in_diagnostics(self) -> bool:
        return self._tendencies_in_diagnostics  # value cannot be modified

    @abc.abstractmethod
    def _call(
        self,
        state: "DataArrayDict",
        timestep: timedelta,
        out_diagnostics: "DataArrayDict",
        out_state: "DataArrayDict",
    ) -> None:
        """TODO."""
        pass
