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
import warnings
from typing import List, Optional, TYPE_CHECKING, Tuple

from sympl._core.base_component import BaseComponent
from sympl._core.combine_properties import (
    combine_component_properties,
    combine_properties,
)
from sympl._core.composite import ImplicitTendencyComponentComposite
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
from sympl._core.state import copy_untouched_quantities
from sympl._core.static_checkers import StaticComponentChecker
from sympl._core.tracers import TracerPacker
from sympl._core.units import clean_units

if TYPE_CHECKING:
    from sympl._core.typingx import (
        DataArrayDict,
        NDArrayLikeDict,
        PropertyDict,
    )


class Stepper(BaseComponent):
    """
    Attributes
    ----------
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    diagnostic_properties : dict
        A dictionary whose keys are quantities for which values
        for the old state are returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    output_properties : dict
        A dictionary whose keys are quantities for which values
        for the new state are returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    tendencies_in_diagnostics : bool
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output based on first order time
        differencing of output values.
    time_unit_name : str
        The unit to use for time differencing when putting tendencies in
        diagnostics.
    time_unit_timedelta: timedelta
        A timedelta corresponding to a single time unit as used for time
        differencing when putting tendencies in diagnostics.
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".
    """

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
        """
        Initializes the Stepper object.

        Args
        ----
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output based on first order time
            differencing of output values.
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
            StaticComponentChecker.factory("diagnostic_properties").check(self)
            StaticComponentChecker.factory("output_properties").check(self)

            self._input_checker = InflowComponentChecker.factory(
                "input_properties"
            )
            self._diagnostic_checker = OutflowComponentChecker.factory(
                "diagnostic_properties"
            )
            self._output_checker = OutflowComponentChecker.factory(
                "output_properties"
            )

            if tendencies_in_diagnostics:
                self._diagnostic_checker.ignored_diagnostics = (
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
            "input_tendencies", self
        )
        self._diagnostic_operator = OutflowComponentOperator.factory(
            "diagnostic_tendencies", self
        )
        self._output_operator = OutflowComponentOperator.factory(
            "output_tendencies", self
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
        self, state: "DataArrayDict", timestep: timedelta
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

        # compute
        raw_diagnostics, raw_new_state = self.array_call(raw_state, timestep)
        if self.uses_tracers:
            new_state = self._tracer_packer.unpack(
                raw_new_state.pop("tracers"), state
            )
        else:
            new_state = {}

        # outflow checks
        if self._enable_checks:
            self._diagnostic_checker.check(raw_diagnostics, state)
            self._output_checker.check_outputs(raw_new_state, state)

        # compute first-order approximation to tendencies
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(
                raw_state, raw_new_state, timestep, raw_diagnostics
            )

        # wrap output arrays in dataarrays
        diagnostics = self._diagnostic_operator.get_dataarray_dict(
            raw_diagnostics, state
        )
        new_state.update(
            self._output_operator.get_dataarray_dict(raw_new_state, state)
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
    def array_call(
        self, state: "NDArrayLikeDict", timestep: timedelta
    ) -> Tuple["NDArrayLikeDict", "NDArrayLikeDict"]:
        """
        Gets diagnostics from the current model state and steps the state
        forward in time according to the timestep.

        Args
        ----
        state : dict
            A numpy array state dictionary. Instead of data arrays, should
            include numpy arrays that satisfy the input_properties of this
            object.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state, as numpy arrays.
        new_state : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the timestep after input state, as numpy arrays.
        """
        pass

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


class TendencyStepper(Stepper):
    """An object which integrates model state forward in time.

    It uses TendencyComponent and DiagnosticComponent objects to update the current model state
    with diagnostics, and to return the model state at the next timestep.

    Attributes
    ----------
    diagnostic_properties : dict
        A dictionary whose keys are quantities for which values
        for the old state are returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    output_properties : dict
        A dictionary whose keys are quantities for which values
        for the new state are returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    prognostic : ImplicitTendencyComponentComposite
        A composite of the TendencyComponent and ImplicitPrognostic objects used by
        the TendencyStepper.
    prognostic_list: list of TendencyComponent and ImplicitPrognosticComponent
        A list of TendencyComponent objects called by the TendencyStepper. These should
        be referenced when determining what inputs are necessary for the
        TendencyStepper.
    tendencies_in_diagnostics : bool
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output.
    time_unit_name : str
        The unit to use for time differencing when putting tendencies in
        diagnostics.
    time_unit_timedelta: timedelta
        A timedelta corresponding to a single time unit as used for time
        differencing when putting tendencies in diagnostics.
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".
    """

    __metaclass__ = abc.ABCMeta

    time_unit_name = "s"
    time_unit_timedelta = timedelta(seconds=1)

    @property
    def input_properties(self):
        input_properties = combine_component_properties(
            self.prognostic_list, "input_properties"
        )
        return combine_properties([input_properties, self.output_properties])

    @property
    def _tendencycomponent_input_properties(self):
        return combine_component_properties(
            self.prognostic_list, "input_properties"
        )

    @property
    def diagnostic_properties(self):
        return_value = {}
        for prognostic in self.prognostic_list:
            return_value.update(prognostic.diagnostic_properties)
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostic_properties(
                return_value, self._tendency_properties
            )
        return return_value

    def _insert_tendencies_to_diagnostic_properties(
        self, diagnostic_properties, tendency_properties
    ):
        for quantity_name, properties in tendency_properties.items():
            tendency_name = self._get_tendency_name(quantity_name)
            diagnostic_properties[tendency_name] = {
                "units": properties["units"],
                "dims": properties["dims"],
            }

    @property
    def output_properties(self):
        output_properties = self._tendency_properties
        for name, properties in output_properties.items():
            properties["units"] += " {}".format(self.time_unit_name)
            properties["units"] = clean_units(properties["units"])
        return output_properties

    @property
    def _tendency_properties(self):
        return_dict = {}
        return_dict.update(
            combine_component_properties(
                self.prognostic_list,
                "tendency_properties",
                input_properties=self._tendencycomponent_input_properties,
            )
        )
        return return_dict

    def __str__(self):
        return (
            "instance of {}(TendencyStepper)\n"
            "    TendencyComponent components: {}".format(self.prognostic_list)
        )

    def __repr__(self):
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

    def array_call(self, state, timestep):
        raise NotImplementedError(
            "TendencyStepper objects do not implement array_call"
        )

    def __init__(self, *args, **kwargs):
        """
        Initialize the TendencyStepper.

        Parameters
        ----------
        *args : TendencyComponent or ImplicitTendencyComponent
            Objects to call for tendencies when doing time stepping.
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output. Default is False. If set to
            True, you probably want to give a name also.
        name : str
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name is used.
        """
        if len(args) == 1 and isinstance(args[0], list):
            warnings.warn(
                "TimeSteppers should be given individual Prognostics rather "
                "than a list, and will not accept lists in a later version.",
                DeprecationWarning,
            )
            args = args[0]
        if any(isinstance(a, ImplicitTendencyComponent) for a in args):
            warnings.warn(
                "Using an ImplicitTendencyComponent in sympl TendencyStepper objects may "
                "lead to scientifically invalid results. Make sure the component "
                "follows the same numerical assumptions as the TendencyStepper used."
            )
        self.prognostic = ImplicitTendencyComponentComposite(*args)
        super(TendencyStepper, self).__init__(**kwargs)
        for name in self.prognostic.tendency_properties.keys():
            if name not in self.output_properties.keys():
                raise InvalidPropertyDictError(
                    "Prognostic object has tendency output for {} but "
                    "TendencyStepper containing it does not have it in "
                    "output_properties.".format(name)
                )
        self.__initialized = True

    @property
    def prognostic_list(self):
        return self.prognostic.component_list

    @property
    def tendencies_in_diagnostics(self):  # value cannot be modified
        return self._tendencies_in_diagnostics

    def _get_tendency_name(self, quantity_name):
        return "{}_tendency_from_{}".format(quantity_name, self.name)

    def __call__(self, state, timestep):
        """
        Retrieves any diagnostics and returns a new state corresponding
        to the next timestep.

        Args
        ----
        state : dict
            The current model state.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state.
        new_state : dict
            The model state at the next timestep.
        """
        if not self.__initialized:
            raise AssertionError(
                "TendencyStepper component has not had its base class "
                "__init__ called, likely due to a missing call to "
                "super(ClassName, self).__init__(*args, **kwargs) in its "
                "__init__ method."
            )
        diagnostics, new_state = self._call(state, timestep)
        copy_untouched_quantities(state, new_state)
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(
                state, new_state, timestep, diagnostics
            )
        return diagnostics, new_state

    def _insert_tendencies_to_diagnostics(
        self, state, new_state, timestep, diagnostics
    ):
        output_properties = self.output_properties
        input_properties = self.input_properties
        for name in output_properties.keys():
            tendency_name = self._get_tendency_name(name)
            if tendency_name in diagnostics.keys():
                raise RuntimeError(
                    "A TendencyComponent has output tendencies as a diagnostic and has"
                    " caused a name clash when trying to do so from this "
                    "TendencyStepper ({}). You must disable "
                    "tendencies_in_diagnostics for this TendencyStepper.".format(
                        tendency_name
                    )
                )
            base_units = input_properties[name]["units"]
            diagnostics[tendency_name] = (
                (
                    new_state[name].to_units(base_units)
                    - state[name].to_units(base_units)
                )
                / timestep.total_seconds()
                * self.time_unit_timedelta.total_seconds()
            )
            if base_units == "":
                diagnostics[tendency_name].attrs["units"] = "{}^-1".format(
                    self.time_unit_name
                )
            else:
                diagnostics[tendency_name].attrs["units"] = "{} {}^-1".format(
                    base_units, self.time_unit_name
                )

    def _call(self, state, timestep):
        """
        Retrieves any diagnostics and returns a new state corresponding
        to the next timestep.

        Args
        ----
        state : dict
            The current model state.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state.
        new_state : dict
            The model state at the next timestep.
        """
        raise NotImplementedError()
