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
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Union

from sympl._core.base_component import BaseComponent
from sympl._core.dynamic_checkers import (
    InflowComponentChecker,
    OutflowComponentChecker,
)
from sympl._core.dynamic_operators import (
    InflowComponentOperator,
    OutflowComponentOperator,
)
from sympl._core.static_checkers import StaticComponentChecker
from sympl._core.time import FakeTimer as Timer

# from sympl._core.time import Timer
from sympl._core.tracers import TracerPacker

if TYPE_CHECKING:
    from datetime import timedelta
    from sympl._core.typingx import (
        DataArrayDict,
        NDArrayLike,
        NDArrayLikeDict,
        PropertyDict,
    )


class DiagnosticComponent(BaseComponent):
    """TODO."""

    name = None

    def __init__(self, *, enable_checks: bool = True) -> None:
        """TODO."""
        super().__init__()

        self.name = self.__class__.__name__
        self._enable_checks = enable_checks

        if enable_checks:
            StaticComponentChecker.factory("input_properties").check(self)
            StaticComponentChecker.factory("diagnostic_properties").check(self)

            self._input_checker = InflowComponentChecker.factory(
                "input_properties", self
            )
            self._diagnostic_inflow_checker = InflowComponentChecker.factory(
                "diagnostic_properties", self
            )
            self._diagnostic_outflow_checker = OutflowComponentChecker.factory(
                "diagnostic_properties", self
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

    def __str__(self) -> str:
        return (
            "instance of {}(DiagnosticComponent)\n"
            "    inputs: {}\n"
            "    diagnostics: {}".format(
                self.__class__,
                self.input_properties.keys(),
                self.diagnostic_properties.keys(),
            )
        )

    def __repr__(self) -> str:
        if hasattr(self, "_making_repr") and self._making_repr:
            return "{}(recursive reference)".format(self.__class__)
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
        self, state: "DataArrayDict", *, out: Optional["DataArrayDict"] = None,
    ) -> "DataArrayDict":
        """TODO."""
        # inflow checks
        Timer.start(label="check")
        if self._enable_checks:
            self._input_checker.check(state)
        Timer.stop()

        # extract raw state
        Timer.start(label="operation")
        raw_state = self._input_operator.get_ndarray_dict(state)
        raw_state["time"] = state["time"]
        Timer.stop()

        # run checks on out
        Timer.start(label="check")
        out = out if out is not None else {}
        if self._enable_checks:
            self._diagnostic_inflow_checker.check(out, state)
        Timer.stop()

        # extract or allocate output buffers
        Timer.start(label="operation")
        raw_diagnostics = self._diagnostic_inflow_operator.get_ndarray_dict(
            out
        )
        raw_diagnostics.update(
            {
                name: self.allocate_diagnostic(name)
                for name in self.diagnostic_properties
                if name not in out
            }
        )
        Timer.stop()

        # run checks on raw_diagnostics
        Timer.start(label="check")
        if self._enable_checks:
            self._diagnostic_outflow_checker.check(raw_diagnostics, state)
        Timer.stop()

        # compute
        Timer.start(label="array_call")
        self.array_call(raw_state, raw_diagnostics)
        Timer.stop()

        # outflow checks
        Timer.start(label="check")
        if self._enable_checks:
            self._diagnostic_outflow_checker.check(raw_diagnostics, state)
        Timer.stop()

        # wrap arrays in dataarrays
        Timer.start(label="operation")
        diagnostics = self._diagnostic_outflow_operator.get_dataarray_dict(
            raw_diagnostics, state, out=out
        )
        Timer.stop()

        return diagnostics

    @property
    @abc.abstractmethod
    def input_properties(self) -> "PropertyDict":
        """TODO."""
        pass

    @property
    @abc.abstractmethod
    def diagnostic_properties(self) -> "PropertyDict":
        """TODO."""
        pass

    @abc.abstractmethod
    def allocate_diagnostic(self, name: str) -> "NDArrayLike":
        """TODO."""
        pass

    @abc.abstractmethod
    def array_call(
        self, state: "NDArrayLikeDict", out: "NDArrayLikeDict"
    ) -> None:
        """TODO."""
        pass


class TendencyComponentUtils:
    @staticmethod
    def init(
        component: Union["ImplicitTendencyComponent", "TendencyComponent"],
        tendencies_in_diagnostics: bool,
        name: str,
        enable_checks: bool,
    ) -> Union["ImplicitTendencyComponent", "TendencyComponent"]:
        component._tendencies_in_diagnostics = tendencies_in_diagnostics
        component.name = name or component.__class__.__name__
        component._enable_checks = enable_checks

        if enable_checks:
            StaticComponentChecker.factory("input_properties").check(component)
            StaticComponentChecker.factory("tendency_properties").check(
                component
            )
            StaticComponentChecker.factory("diagnostic_properties").check(
                component
            )

            component._input_checker = InflowComponentChecker.factory(
                "input_properties", component
            )
            component._tendency_inflow_checker = InflowComponentChecker.factory(
                "tendency_properties", component
            )
            component._tendency_outflow_checker = OutflowComponentChecker.factory(
                "tendency_properties", component
            )
            component._diagnostic_inflow_checker = InflowComponentChecker.factory(
                "diagnostic_properties", component
            )
            component._diagnostic_outflow_checker = OutflowComponentChecker.factory(
                "diagnostic_properties", component
            )

        if component.tendencies_in_diagnostics:
            component._added_diagnostic_names = (
                component._insert_tendency_properties()
            )
            if enable_checks:
                component._diagnostic_inflow_checker.ignored_diagnostics = (
                    component._added_diagnostic_names
                )
                component._diagnostic_outflow_checker.ignored_diagnostics = (
                    component._added_diagnostic_names
                )
        else:
            component._added_diagnostic_names = []

        if component.uses_tracers:
            if component.tracer_dims is None:
                raise ValueError(
                    f"Component of type {component.__class__.__name__} must "
                    f"specify tracer_dims property when uses_tracers=True."
                )
            prepend_tracers = getattr(component, "prepend_tracers", None)
            component._tracer_packer = TracerPacker(
                component,
                component.tracer_dims,
                prepend_tracers=prepend_tracers,
            )

        component._input_operator = InflowComponentOperator.factory(
            "input_properties", component
        )
        component._tendency_inflow_operator = InflowComponentOperator.factory(
            "tendency_properties", component
        )
        component._tendency_outflow_operator = OutflowComponentOperator.factory(
            "tendency_properties", component
        )
        component._diagnostic_inflow_operator = InflowComponentOperator.factory(
            "diagnostic_properties", component
        )
        component._diagnostic_outflow_operator = OutflowComponentOperator.factory(
            "diagnostic_properties", component
        )

        return component

    @staticmethod
    def preprocessing(
        component: ["ImplicitTendencyComponent", "TendencyComponent"],
        state: "DataArrayDict",
        out_tendencies: Optional["DataArrayDict"],
        out_diagnostics: Optional["DataArrayDict"],
        overwrite_tendencies: Optional[Dict[str, bool]],
    ) -> Tuple[
        "NDArrayLikeDict",
        "NDArrayLikeDict",
        "NDArrayLikeDict",
        Dict[str, bool],
    ]:
        # inflow checks
        Timer.start(label="check")
        if component._enable_checks:
            component._input_checker.check(state)
        Timer.stop()

        # extract raw state
        Timer.start(label="operation")
        raw_state = component._input_operator.get_ndarray_dict(state)
        if component.uses_tracers:
            raw_state["tracers"] = component._tracer_packer.pack(state)
        raw_state["time"] = state["time"]
        Timer.stop()

        # run checks on out_tendencies
        Timer.start(label="check")
        out_tendencies = out_tendencies if out_tendencies is not None else {}
        if component._enable_checks:
            component._tendency_inflow_checker.check(out_tendencies, state)
        Timer.stop()

        # extract or allocate buffers for tendencies
        Timer.start(label="operation")
        raw_tendencies = component._tendency_inflow_operator.get_ndarray_dict(
            out_tendencies
        )
        raw_tendencies.update(
            {
                name: component.allocate_tendency(name)
                for name in component.tendency_properties
                if name not in out_tendencies
            }
        )
        Timer.stop()

        # run checks on raw_tendencies
        Timer.start(label="check")
        if component._enable_checks:
            component._tendency_outflow_checker.check(raw_tendencies, state)
        Timer.stop()

        # set overwrite_tendencies
        Timer.start(label="operation")
        overwrite_tendencies = overwrite_tendencies or {}
        for name in component.tendency_properties:
            overwrite_tendencies.setdefault(name, True)
        Timer.stop()

        # run checks on out
        Timer.start(label="check")
        out_diagnostics = (
            out_diagnostics if out_diagnostics is not None else {}
        )
        if component._enable_checks:
            component._diagnostic_inflow_checker.check(out_diagnostics, state)
        Timer.stop()

        # extract or allocate output buffers
        Timer.start(label="operation")
        raw_diagnostics = component._diagnostic_inflow_operator.get_ndarray_dict(
            out_diagnostics
        )
        raw_diagnostics.update(
            {
                name: component.allocate_diagnostic(name)
                for name in component.diagnostic_properties
                if name not in out_diagnostics
            }
        )
        Timer.stop()

        # run checks on raw_diagnostics
        Timer.start(label="check")
        if component._enable_checks:
            component._diagnostic_outflow_checker.check(raw_diagnostics, state)
        Timer.stop()

        return raw_state, raw_tendencies, raw_diagnostics, overwrite_tendencies

    @staticmethod
    def postprocessing(
        component: Union["ImplicitTendencyComponent", "TendencyComponent"],
        state: "DataArrayDict",
        out_tendencies: Optional["DataArrayDict"],
        out_diagnostics: Optional["DataArrayDict"],
        raw_tendencies: "NDArrayLikeDict",
        raw_diagnostics: "NDArrayLikeDict",
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        # # process tracers
        # if component.uses_tracers:
        #     out_tendencies = component._tracer_packer.unpack(
        #         raw_tendencies.pop("tracers"),
        #         state,
        #         multiply_unit=component.tracer_tendency_time_unit,
        #     )
        # else:
        #     out_tendencies = {}

        # outflow checks
        Timer.start(label="check")
        if component._enable_checks:
            component._tendency_outflow_checker.check(raw_tendencies, state)
            component._diagnostic_outflow_checker.check(raw_diagnostics, state)
        Timer.stop()

        # wrap arrays in dataarrays
        Timer.start(label="operation")
        tendencies = component._tendency_outflow_operator.get_dataarray_dict(
            raw_tendencies, state, out=out_tendencies
        )
        diagnostics = component._diagnostic_outflow_operator.get_dataarray_dict(
            raw_diagnostics, state, out=out_diagnostics
        )
        if component.tendencies_in_diagnostics:
            component._insert_tendencies_to_diagnostics(
                tendencies, diagnostics
            )
        component._last_update_time = state["time"]
        Timer.stop()

        return tendencies, diagnostics


class ImplicitTendencyComponent(BaseComponent):
    """TODO."""

    name = None
    uses_tracers = False
    tracer_dims = None
    tracer_tendency_time_unit = "s^-1"

    def __init__(
        self,
        tendencies_in_diagnostics: bool = False,
        name: Optional[str] = None,
        *,
        enable_checks: bool = True
    ) -> None:
        """TODO."""
        super().__init__()
        TendencyComponentUtils.init(
            self, tendencies_in_diagnostics, name, enable_checks
        )

    def __str__(self) -> str:
        return (
            f"Instance of {self.__class__.__name__}(TendencyComponent)\n"
            f"    inputs: {', '.join(self.input_properties.keys())}\n"
            f"    tendencies: {', '.join(self.tendency_properties.keys())}\n"
            f"    diagnostics: {', '.join(self.diagnostic_properties)}"
        )

    def __repr__(self) -> str:
        if hasattr(self, "_making_repr") and self._making_repr:
            return "{}(recursive reference)".format(self.__class__)
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
        timestep: "timedelta",
        *,
        out_tendencies: Optional["DataArrayDict"] = None,
        out_diagnostics: Optional["DataArrayDict"] = None,
        overwrite_tendencies: Optional[Dict[str, bool]] = None,
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        """TODO."""
        # pre-processing
        (
            raw_state,
            raw_tendencies,
            raw_diagnostics,
            overwrite_tendencies,
        ) = TendencyComponentUtils.preprocessing(
            self, state, out_tendencies, out_diagnostics, overwrite_tendencies
        )

        # compute
        Timer.start(label="array_call")
        self.array_call(
            raw_state,
            timestep,
            raw_tendencies,
            raw_diagnostics,
            overwrite_tendencies,
        )
        Timer.stop()

        # post-processing
        tendencies, diagnostics = TendencyComponentUtils.postprocessing(
            self,
            state,
            out_tendencies,
            out_diagnostics,
            raw_tendencies,
            raw_diagnostics,
        )

        return tendencies, diagnostics

    @property
    def tendencies_in_diagnostics(self) -> bool:
        """TODO."""
        return self._tendencies_in_diagnostics

    @property
    @abc.abstractmethod
    def input_properties(self) -> "PropertyDict":
        """TODO."""
        pass

    @property
    @abc.abstractmethod
    def tendency_properties(self) -> "PropertyDict":
        """TODO."""
        pass

    @property
    @abc.abstractmethod
    def diagnostic_properties(self) -> "PropertyDict":
        """TODO."""
        pass

    @abc.abstractmethod
    def allocate_tendency(self, name) -> "NDArrayLike":
        """TODO."""
        pass

    @abc.abstractmethod
    def allocate_diagnostic(self, name) -> "NDArrayLike":
        """TODO."""
        pass

    @abc.abstractmethod
    def array_call(
        self,
        state: "NDArrayLikeDict",
        timestep: "timedelta",
        out_tendencies: "NDArrayLikeDict",
        out_diagnostics: "NDArrayLikeDict",
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        """TODO."""
        pass

    def _insert_tendency_properties(self) -> List[str]:
        added_names = []
        for name, properties in self.tendency_properties.items():
            tendency_name = self._get_tendency_name(name)
            if "dims" in properties.keys():
                dims = properties["dims"]
            else:
                dims = self.input_properties[name]["dims"]
            self.diagnostic_properties[tendency_name] = {
                "units": properties["units"],
                "dims": dims,
            }
            added_names.append(tendency_name)
        return added_names

    def _get_tendency_name(self, name: str) -> str:
        return f"{name}_tendency_from_{self.name}"

    def _insert_tendencies_to_diagnostics(
        self, tendencies: "DataArrayDict", diagnostics: "DataArrayDict"
    ) -> None:
        for name, value in tendencies.items():
            tendency_name = self._get_tendency_name(name)
            diagnostics[tendency_name] = value


class TendencyComponent(BaseComponent):
    """TODO."""

    name = None
    uses_tracers = False
    tracer_dims = None
    tracer_tendency_time_unit = "s^-1"

    def __init__(
        self,
        tendencies_in_diagnostics: bool = False,
        name: Optional[str] = None,
        *,
        enable_checks: bool = True
    ) -> None:
        """TODO."""
        super().__init__()
        TendencyComponentUtils.init(
            self, tendencies_in_diagnostics, name, enable_checks
        )

    def __str__(self) -> str:
        return (
            f"Instance of {self.__class__.__name__}(TendencyComponent)\n"
            f"    inputs: {', '.join(self.input_properties.keys())}\n"
            f"    tendencies: {', '.join(self.tendency_properties.keys())}\n"
            f"    diagnostics: {', '.join(self.diagnostic_properties.keys())}"
        )

    def __repr__(self) -> str:
        if hasattr(self, "_making_repr") and self._making_repr:
            return "{}(recursive reference)".format(self.__class__)
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
        *,
        out_tendencies: Optional["DataArrayDict"] = None,
        out_diagnostics: Optional["DataArrayDict"] = None,
        overwrite_tendencies: Optional[Dict[str, bool]] = None,
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        """TODO."""
        # pre-processing
        (
            raw_state,
            raw_tendencies,
            raw_diagnostics,
            overwrite_tendencies,
        ) = TendencyComponentUtils.preprocessing(
            self, state, out_tendencies, out_diagnostics, overwrite_tendencies
        )

        # compute
        Timer.start(label="array_call")
        self.array_call(
            raw_state, raw_tendencies, raw_diagnostics, overwrite_tendencies,
        )
        Timer.stop()

        # post-processing
        tendencies, diagnostics = TendencyComponentUtils.postprocessing(
            self,
            state,
            out_tendencies,
            out_diagnostics,
            raw_tendencies,
            raw_diagnostics,
        )

        return tendencies, diagnostics

    @property
    def tendencies_in_diagnostics(self) -> bool:
        """TODO."""
        return self._tendencies_in_diagnostics

    @property
    @abc.abstractmethod
    def input_properties(self) -> "PropertyDict":
        """TODO."""
        pass

    @property
    @abc.abstractmethod
    def tendency_properties(self) -> "PropertyDict":
        """TODO."""
        pass

    @property
    @abc.abstractmethod
    def diagnostic_properties(self) -> "PropertyDict":
        """TODO."""
        pass

    @abc.abstractmethod
    def allocate_tendency(self, name) -> "NDArrayLike":
        pass

    @abc.abstractmethod
    def allocate_diagnostic(self, name) -> "NDArrayLike":
        """TODO."""
        pass

    @abc.abstractmethod
    def array_call(
        self,
        state: "NDArrayLikeDict",
        out_tendencies: "NDArrayLikeDict",
        out_diagnostics: "NDArrayLikeDict",
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        """TODO."""
        pass

    def _insert_tendency_properties(self) -> List[str]:
        added_names = []
        for name, properties in self.tendency_properties.items():
            tendency_name = self._get_tendency_name(name)
            if "dims" in properties.keys():
                dims = properties["dims"]
            else:
                dims = self.input_properties[name]["dims"]
            self.diagnostic_properties[tendency_name] = {
                "units": properties["units"],
                "dims": dims,
            }
            added_names.append(tendency_name)
        return added_names

    def _get_tendency_name(self, name: str) -> str:
        return f"{name}_tendency_from_{self.name}"

    def _insert_tendencies_to_diagnostics(
        self, tendencies: "DataArrayDict", diagnostics: "DataArrayDict"
    ) -> None:
        for name, value in tendencies.items():
            tendency_name = self._get_tendency_name(name)
            diagnostics[tendency_name] = value
