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
from typing import List, Optional, TYPE_CHECKING, Tuple

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
from sympl._core.tracers import TracerPacker

if TYPE_CHECKING:
    from datetime import timedelta
    from sympl._core.typingx import (
        DataArrayDict,
        NDArrayLikeDict,
        PropertyDict,
    )


class DiagnosticComponent(BaseComponent):
    """
    Attributes
    ----------
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    diagnostic_properties : dict
        A dictionary whose keys are diagnostic quantities returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    """

    def __init__(self, *, enable_checks: bool = True) -> None:
        super().__init__()

        self._enable_checks = enable_checks

        if enable_checks:
            StaticComponentChecker.factory("input_properties").check(self)
            StaticComponentChecker.factory("diagnostic_properties").check(self)

            self._input_checker = InflowComponentChecker.factory(
                "input_properties", self
            )
            self._diagnostic_checker = OutflowComponentChecker.factory(
                "diagnostic_properties", self
            )

        self._input_operator = InflowComponentOperator.factory(
            "input_properties", self
        )
        self._diagnostic_operator = OutflowComponentOperator.factory(
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

    def __call__(self, state: "DataArrayDict") -> "DataArrayDict":
        """
        Gets diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary satisfying the input_properties of this
            object.

        Returns
        -------
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state.

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the TendencyComponent instance.
        """
        # inflow checks
        if self._enable_checks:
            self._input_checker.check(state)

        # extract raw state
        raw_state = self._input_operator.get_ndarray_dict(state)
        raw_state["time"] = state["time"]

        # compute
        raw_diagnostics = self.array_call(raw_state)

        # outflow checks
        if self._enable_checks:
            self._diagnostic_checker.check(raw_diagnostics, state)

        # wrap arrays in dataarrays
        diagnostics = self._diagnostic_operator.get_dataarray_dict(
            raw_diagnostics, state
        )

        return diagnostics

    @property
    @abc.abstractmethod
    def input_properties(self) -> "PropertyDict":
        pass

    @property
    @abc.abstractmethod
    def diagnostic_properties(self) -> "PropertyDict":
        pass

    @abc.abstractmethod
    def array_call(self, state: "NDArrayLikeDict") -> "NDArrayLikeDict":
        """
        Gets diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary. Instead of data arrays, should
            include numpy arrays that satisfy the input_properties of this
            object.

        Returns
        -------
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state, as numpy arrays.
        """
        pass


class ImplicitTendencyComponent(BaseComponent):
    """
    Attributes
    ----------
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    tendency_properties : dict
        A dictionary whose keys are quantities for which tendencies are returned
        when the object is called, and values are dictionaries which indicate
        'dims' and 'units'.
    diagnostic_properties : dict
        A dictionary whose keys are diagnostic quantities returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    tendencies_in_diagnostics : bool
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output based on first order time
        differencing of output values.
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".
    """

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
        """
        Initializes the Stepper object.

        Args
        ----
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : string, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        """
        super().__init__()

        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__
        self._enable_checks = enable_checks

        if enable_checks:
            StaticComponentChecker.factory("input_properties").check(self)
            StaticComponentChecker.factory("tendency_properties").check(self)
            StaticComponentChecker.factory("diagnostic_properties").check(self)

            self._input_checker = InflowComponentChecker.factory(
                "input_properties", self
            )
            self._tendency_checker = OutflowComponentChecker.factory(
                "tendency_properties", self
            )
            self._diagnostic_checker = OutflowComponentChecker.factory(
                "diagnostic_properties", self
            )

        if self.tendencies_in_diagnostics:
            self._added_diagnostic_names = self._insert_tendency_properties()
            if enable_checks:
                self._diagnostic_checker.ignored_diagnostics = (
                    self._added_diagnostic_names
                )
        else:
            self._added_diagnostic_names = []

        if self.uses_tracers:
            if self.tracer_dims is None:
                raise ValueError(
                    f"Component of type {self.__class__.__name__} must "
                    f"specify tracer_dims property when uses_tracers=True."
                )
            prepend_tracers = getattr(self, "prepend_tracers", None)
            self._tracer_packer = TracerPacker(
                self, self.tracer_dims, prepend_tracers=prepend_tracers
            )

        self._input_operator = InflowComponentOperator.factory(
            "input_properties", self
        )
        self._tendency_operator = OutflowComponentOperator.factory(
            "tendency_properties", self
        )
        self._diagnostic_operator = OutflowComponentOperator.factory(
            "diagnostic_properties", self
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
        self, state: "DataArrayDict", timestep: "timedelta"
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary satisfying the input_properties of this
            object.
        timestep : timedelta
            The time over which the model is being stepped.

        Returns
        -------
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second at the time of the input state.

        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state.

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the TendencyComponent instance.
        """
        # inflow checks
        if self._enable_checks:
            self._input_checker.check(state)

        # extract raw state
        raw_state = self._input_operator.get_ndarray_dict(state)
        if self.uses_tracers:
            raw_state["tracers"] = self._tracer_packer.pack(state)
        raw_state["time"] = state["time"]

        # compute
        raw_tendencies, raw_diagnostics = self.array_call(raw_state, timestep)

        # process tracers
        if self.uses_tracers:
            out_tendencies = self._tracer_packer.unpack(
                raw_tendencies.pop("tracers"),
                state,
                multiply_unit=self.tracer_tendency_time_unit,
            )
        else:
            out_tendencies = {}

        # outflow checks
        if self._enable_checks:
            self._tendency_checker.check(raw_tendencies, state)
            self._diagnostic_checker.check(raw_diagnostics, state)

        # wrap arrays in dataarrays
        out_tendencies.update(
            self._tendency_operator.get_dataarray_dict(raw_tendencies, state)
        )
        diagnostics = self._diagnostic_operator.get_dataarray_dict(
            raw_diagnostics, state
        )
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(out_tendencies, diagnostics)
        self._last_update_time = state["time"]

        return out_tendencies, diagnostics

    @property
    def tendencies_in_diagnostics(self) -> bool:
        return self._tendencies_in_diagnostics

    @property
    @abc.abstractmethod
    def input_properties(self) -> "PropertyDict":
        pass

    @property
    @abc.abstractmethod
    def tendency_properties(self) -> "PropertyDict":
        pass

    @property
    @abc.abstractmethod
    def diagnostic_properties(self) -> "PropertyDict":
        pass

    @abc.abstractmethod
    def array_call(
        self, state: "NDArrayLikeDict", timestep: "timedelta"
    ) -> Tuple["NDArrayLikeDict", "NDArrayLikeDict"]:
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary. Instead of data arrays, should
            include numpy arrays that satisfy the input_properties of this
            object.
        timestep : timedelta
            The time over which the model is being stepped.

        Returns
        -------
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second at the time of the input state, as
            numpy arrays.

        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state, as numpy arrays.
        """
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
    """
    Attributes
    ----------
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    tendency_properties : dict
        A dictionary whose keys are quantities for which tendencies are returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    diagnostic_properties : dict
        A dictionary whose keys are diagnostic quantities returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    tendencies_in_diagnostics : bool
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output based on first order time
        differencing of output values.
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".
    """

    name = None
    uses_tracers = False
    tracer_dims = None
    tracer_tendency_time_unit = "s^-1"

    def __init__(
        self,
        tendencies_in_diagnostics: bool = False,
        name: Optional[str] = None,
        *,
        enable_checks: bool = False
    ) -> None:
        """
        Initializes the Stepper object.

        Args
        ----
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : string, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        """
        super().__init__()

        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__
        self._enable_checks = enable_checks

        if enable_checks:
            StaticComponentChecker.factory("input_properties").check(self)
            StaticComponentChecker.factory("tendency_properties").check(self)
            StaticComponentChecker.factory("diagnostic_properties").check(self)

            self._input_checker = InflowComponentChecker.factory(
                "input_properties", self
            )
            self._tendency_checker = OutflowComponentChecker.factory(
                "tendency_properties", self
            )
            self._diagnostic_checker = OutflowComponentChecker.factory(
                "diagnostic_properties", self
            )

        if tendencies_in_diagnostics:
            self._added_diagnostic_names = self._insert_tendency_properties()
            if enable_checks:
                self._diagnostic_checker.ignored_diagnostics = (
                    self._added_diagnostic_names
                )
        else:
            self._added_diagnostic_names = []

        if self.uses_tracers:
            if self.tracer_dims is None:
                raise ValueError(
                    f"Component of type {self.__class__.__name__} must "
                    f"specify tracer_dims property when uses_tracers=True."
                )
            prepend_tracers = getattr(self, "prepend_tracers", None)
            self._tracer_packer = TracerPacker(
                self, self.tracer_dims, prepend_tracers=prepend_tracers
            )

        self._input_operator = InflowComponentOperator.factory(
            "input_properties", self
        )
        self._tendency_operator = OutflowComponentOperator.factory(
            "tendency_properties", self
        )
        self._diagnostic_operator = OutflowComponentOperator.factory(
            "diagnostic_properties", self
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
        self, state: "DataArrayDict"
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary satisfying the input_properties of this
            object.

        Returns
        -------
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second at the time of the input state.

        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state.

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the TendencyComponent instance.
        """
        # inflow checks
        if self._enable_checks:
            self._input_checker.check(state)

        # extract raw state
        raw_state = self._input_operator.get_ndarray_dict(state)
        if self.uses_tracers:
            raw_state["tracers"] = self._tracer_packer.pack(state)
        raw_state["time"] = state["time"]

        # compute
        raw_tendencies, raw_diagnostics = self.array_call(raw_state)

        # process tracers
        if self.uses_tracers:
            out_tendencies = self._tracer_packer.unpack(
                raw_tendencies.pop("tracers"),
                state,
                multiply_unit=self.tracer_tendency_time_unit,
            )
        else:
            out_tendencies = {}

        # outflow checks
        if self._enable_checks:
            self._tendency_checker.check(raw_tendencies, state)
            self._diagnostic_checker.check(raw_diagnostics, state)

        # wrap raw arrays into dataarrays
        out_tendencies.update(
            self._tendency_operator.get_dataarray_dict(raw_tendencies, state)
        )
        diagnostics = self._diagnostic_operator.get_dataarray_dict(
            raw_diagnostics, state
        )
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(out_tendencies, diagnostics)

        return out_tendencies, diagnostics

    @property
    def tendencies_in_diagnostics(self) -> bool:
        return self._tendencies_in_diagnostics

    @property
    @abc.abstractmethod
    def input_properties(self) -> "PropertyDict":
        pass

    @property
    @abc.abstractmethod
    def tendency_properties(self) -> "PropertyDict":
        pass

    @property
    @abc.abstractmethod
    def diagnostic_properties(self) -> "PropertyDict":
        pass

    @abc.abstractmethod
    def array_call(
        self, state: "NDArrayLikeDict"
    ) -> Tuple["NDArrayLikeDict", "NDArrayLikeDict"]:
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary. Instead of data arrays, should
            include numpy arrays that satisfy the input_properties of this
            object.

        Returns
        -------
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second at the time of the input state, as
            numpy arrays.

        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state, as numpy arrays.
        """
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
