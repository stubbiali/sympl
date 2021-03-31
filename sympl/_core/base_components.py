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
from six import add_metaclass

try:
    from inspect import getfullargspec as getargspec
except ImportError:
    from inspect import getargspec

from sympl._core.checks import (
    InputChecker,
    TendencyChecker,
    DiagnosticChecker,
    OutputChecker,
)
from sympl._core.exceptions import InvalidPropertyDictError
from sympl._core.restore_dataarray import restore_data_arrays_with_properties
from sympl._core.tracers import TracerPacker
from sympl._core.utils import get_kwarg_defaults
from sympl._core.storage import get_arrays_with_properties


def is_component_class(cls):
    return any(
        issubclass(cls, cls2)
        for cls2 in (
            Stepper,
            TendencyComponent,
            ImplicitTendencyComponent,
            DiagnosticComponent,
        )
    )


def is_component_base_class(cls):
    return cls in (
        Stepper,
        TendencyComponent,
        ImplicitTendencyComponent,
        DiagnosticComponent,
    )


class ComponentMeta(abc.ABCMeta):
    def __instancecheck__(cls, instance):
        if is_component_class(
            instance.__class__
        ) or not is_component_base_class(cls):
            return issubclass(instance.__class__, cls)
        else:  # checking if non-inheriting instance is a duck-type of a component base class
            (
                required_attributes,
                disallowed_attributes,
            ) = cls.__get_attribute_requirements()
            has_attributes = all(
                hasattr(instance, att) for att in required_attributes
            ) and not any(
                hasattr(instance, att) for att in disallowed_attributes
            )
            if hasattr(cls, "__call__") and not hasattr(instance, "__call__"):
                return False
            elif hasattr(cls, "__call__"):
                timestep_in_class_call = (
                    "timestep" in getargspec(cls.__call__).args
                )
                instance_argspec = getargspec(instance.__call__)
                timestep_in_instance_call = "timestep" in instance_argspec.args
                instance_defaults = get_kwarg_defaults(instance.__call__)
                timestep_is_optional = (
                    "timestep" in instance_defaults.keys()
                    and instance_defaults["timestep"] is None
                )
                has_correct_spec = (
                    timestep_in_class_call == timestep_in_instance_call
                ) or timestep_is_optional
            else:
                raise RuntimeError(
                    "Cannot check instance type on component subclass that has "
                    "no __call__ method"
                )
            return has_attributes and has_correct_spec

    def __get_attribute_requirements(cls):
        check_attributes = (
            "input_properties",
            "tendency_properties",
            "diagnostic_properties",
            "output_properties",
            "__call__",
            "array_call",
            "tendencies_in_diagnostics",
            "name",
        )
        required_attributes = list(
            att for att in check_attributes if hasattr(cls, att)
        )
        disallowed_attributes = list(
            att for att in check_attributes if att not in required_attributes
        )
        if "name" in disallowed_attributes:  # name is always allowed
            disallowed_attributes.remove("name")
        return required_attributes, disallowed_attributes


@add_metaclass(ComponentMeta)
class Stepper(object):
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

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    @abc.abstractproperty
    def output_properties(self):
        return {}

    def __str__(self):
        return (
            "instance of {}(Stepper)\n"
            "    inputs: {}\n"
            "    outputs: {}\n"
            "    diagnostics: {}".format(
                self.__class__,
                self.input_properties.keys(),
                self.output_properties.keys(),
                self.diagnostic_properties.keys(),
            )
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

    def __init__(self, tendencies_in_diagnostics=False, name=None):
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
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__
        super(Stepper, self).__init__()
        self._input_checker = InputChecker(self)
        self._diagnostic_checker = DiagnosticChecker(self)
        self._output_checker = OutputChecker(self)
        if tendencies_in_diagnostics:
            self._diagnostic_checker.set_ignored_diagnostics(
                self._insert_tendency_properties()
            )
        self.__initialized = True
        if self.uses_tracers:
            if self.tracer_dims is None:
                raise ValueError(
                    "Component of type {} must specify tracer_dims property "
                    "when uses_tracers=True".format(self.__class__.__name__)
                )
            prepend_tracers = getattr(self, "prepend_tracers", None)
            self._tracer_packer = TracerPacker(
                self, self.tracer_dims, prepend_tracers=prepend_tracers
            )
        super(Stepper, self).__init__()

    def _insert_tendency_properties(self):
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

    def _get_tendency_name(self, name):
        return "{}_tendency_from_{}".format(name, self.name)

    @property
    def tendencies_in_diagnostics(self):
        return self._tendencies_in_diagnostics  # value cannot be modified

    def _check_self_is_initialized(self):
        try:
            initialized = self.__initialized
        except AttributeError:
            initialized = False
        if not initialized:
            raise RuntimeError(
                "Component has not called __init__ of base class, likely "
                "because its class {} is missing a call to "
                "super({}, self).__init__(**kwargs) in its __init__ "
                "method.".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )

    def __call__(self, state, timestep):
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
        self._check_self_is_initialized()
        self._input_checker.check_inputs(state)
        raw_state = get_arrays_with_properties(state, self.input_properties)
        if self.uses_tracers:
            raw_state["tracers"] = self._tracer_packer.pack(state)
        raw_state["time"] = state["time"]
        raw_diagnostics, raw_new_state = self.array_call(raw_state, timestep)
        if self.uses_tracers:
            new_state = self._tracer_packer.unpack(
                raw_new_state.pop("tracers"), state
            )
        else:
            new_state = {}
        self._diagnostic_checker.check_diagnostics(raw_diagnostics)
        self._output_checker.check_outputs(raw_new_state)
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(
                raw_state, raw_new_state, timestep, raw_diagnostics
            )
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics,
            self.diagnostic_properties,
            state,
            self.input_properties,
        )
        new_state.update(
            restore_data_arrays_with_properties(
                raw_new_state,
                self.output_properties,
                state,
                self.input_properties,
            )
        )
        return diagnostics, new_state

    def _insert_tendencies_to_diagnostics(
        self, raw_state, raw_new_state, timestep, raw_diagnostics
    ):
        for name in self.output_properties.keys():
            tendency_name = self._get_tendency_name(name)
            raw_diagnostics[tendency_name] = (
                (raw_new_state[name] - raw_state[name])
                / timestep.total_seconds()
                * self.time_unit_timedelta.total_seconds()
            )

    @abc.abstractmethod
    def array_call(self, state, timestep):
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


@add_metaclass(ComponentMeta)
class TendencyComponent(object):
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

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def tendency_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    name = None
    uses_tracers = False
    tracer_tendency_time_unit = "s^-1"

    def __str__(self):
        return (
            "instance of {}(TendencyComponent)\n"
            "    inputs: {}\n"
            "    tendencies: {}\n"
            "    diagnostics: {}".format(
                self.__class__,
                self.input_properties.keys(),
                self.tendency_properties.keys(),
                self.diagnostic_properties.keys(),
            )
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

    def __init__(self, tendencies_in_diagnostics=False, name=None):
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
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__
        self._input_checker = InputChecker(self)
        self._tendency_checker = TendencyChecker(self)
        self._diagnostic_checker = DiagnosticChecker(self)
        if self.tendencies_in_diagnostics:
            self._added_diagnostic_names = self._insert_tendency_properties()
            self._diagnostic_checker.set_ignored_diagnostics(
                self._added_diagnostic_names
            )
        else:
            self._added_diagnostic_names = []
        if self.uses_tracers:
            if self.tracer_dims is None:
                raise ValueError(
                    "Component of type {} must specify tracer_dims property "
                    "when uses_tracers=True".format(self.__class__.__name__)
                )
            prepend_tracers = getattr(self, "prepend_tracers", None)
            self._tracer_packer = TracerPacker(
                self, self.tracer_dims, prepend_tracers=prepend_tracers
            )
        self.__initialized = True
        super(TendencyComponent, self).__init__()

    @property
    def tendencies_in_diagnostics(self):
        return self._tendencies_in_diagnostics

    def _insert_tendency_properties(self):
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

    def _get_tendency_name(self, name):
        return "{}_tendency_from_{}".format(name, self.name)

    def _check_self_is_initialized(self):
        try:
            initialized = self.__initialized
        except AttributeError:
            initialized = False
        if not initialized:
            raise RuntimeError(
                "Component has not called __init__ of base class, likely "
                "because its class {} is missing a call to "
                "super({}, self).__init__(**kwargs) in its __init__ "
                "method.".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )

    def __call__(self, state):
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
        self._check_self_is_initialized()
        self._input_checker.check_inputs(state)
        raw_state = get_arrays_with_properties(state, self.input_properties)
        if self.uses_tracers:
            raw_state["tracers"] = self._tracer_packer.pack(state)
        raw_state["time"] = state["time"]
        raw_tendencies, raw_diagnostics = self.array_call(raw_state)
        if self.uses_tracers:
            out_tendencies = self._tracer_packer.unpack(
                raw_tendencies.pop("tracers"),
                state,
                multiply_unit=self.tracer_tendency_time_unit,
            )
        else:
            out_tendencies = {}
        self._tendency_checker.check_tendencies(raw_tendencies)
        self._diagnostic_checker.check_diagnostics(raw_diagnostics)
        out_tendencies.update(
            restore_data_arrays_with_properties(
                raw_tendencies,
                self.tendency_properties,
                state,
                self.input_properties,
            )
        )
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics,
            self.diagnostic_properties,
            state,
            self.input_properties,
            ignore_names=self._added_diagnostic_names,
        )
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(out_tendencies, diagnostics)
        return out_tendencies, diagnostics

    def _insert_tendencies_to_diagnostics(self, tendencies, diagnostics):
        for name, value in tendencies.items():
            tendency_name = self._get_tendency_name(name)
            diagnostics[tendency_name] = value

    @abc.abstractmethod
    def array_call(self, state):
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


@add_metaclass(ComponentMeta)
class ImplicitTendencyComponent(object):
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

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def tendency_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    name = None
    uses_tracers = False
    tracer_tendency_time_unit = "s^-1"

    def __str__(self):
        return (
            "instance of {}(TendencyComponent)\n"
            "    inputs: {}\n"
            "    tendencies: {}\n"
            "    diagnostics: {}".format(
                self.__class__,
                self.input_properties.keys(),
                self.tendency_properties.keys(),
                self.diagnostic_properties.keys(),
            )
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

    def __init__(self, tendencies_in_diagnostics=False, name=None):
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
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__
        self._added_diagnostic_names = []
        self._input_checker = InputChecker(self)
        self._diagnostic_checker = DiagnosticChecker(self)
        self._tendency_checker = TendencyChecker(self)
        if self.tendencies_in_diagnostics:
            self._added_diagnostic_names = self._insert_tendency_properties()
            self._diagnostic_checker.set_ignored_diagnostics(
                self._added_diagnostic_names
            )
        if self.uses_tracers:
            if self.tracer_dims is None:
                raise ValueError(
                    "Component of type {} must specify tracer_dims property "
                    "when uses_tracers=True".format(self.__class__.__name__)
                )
            prepend_tracers = getattr(self, "prepend_tracers", None)
            self._tracer_packer = TracerPacker(
                self, self.tracer_dims, prepend_tracers=prepend_tracers
            )
        self.__initialized = True
        super(ImplicitTendencyComponent, self).__init__()

    @property
    def tendencies_in_diagnostics(self):
        return self._tendencies_in_diagnostics

    def _insert_tendency_properties(self):
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

    def _get_tendency_name(self, name):
        return "{}_tendency_from_{}".format(name, self.name)

    def _check_self_is_initialized(self):
        try:
            initialized = self.__initialized
        except AttributeError:
            initialized = False
        if not initialized:
            raise RuntimeError(
                "Component has not called __init__ of base class, likely "
                "because its class {} is missing a call to "
                "super({}, self).__init__(**kwargs) in its __init__ "
                "method.".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )

    def __call__(self, state, timestep):
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
        self._check_self_is_initialized()
        self._input_checker.check_inputs(state)
        raw_state = get_arrays_with_properties(state, self.input_properties)
        if self.uses_tracers:
            raw_state["tracers"] = self._tracer_packer.pack(state)
        raw_state["time"] = state["time"]
        raw_tendencies, raw_diagnostics = self.array_call(raw_state, timestep)
        if self.uses_tracers:
            out_tendencies = self._tracer_packer.unpack(
                raw_tendencies.pop("tracers"),
                state,
                multiply_unit=self.tracer_tendency_time_unit,
            )
        else:
            out_tendencies = {}
        self._tendency_checker.check_tendencies(raw_tendencies)
        self._diagnostic_checker.check_diagnostics(raw_diagnostics)
        out_tendencies.update(
            restore_data_arrays_with_properties(
                raw_tendencies,
                self.tendency_properties,
                state,
                self.input_properties,
            )
        )
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics,
            self.diagnostic_properties,
            state,
            self.input_properties,
            ignore_names=self._added_diagnostic_names,
        )
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(out_tendencies, diagnostics)
        self._last_update_time = state["time"]
        return out_tendencies, diagnostics

    def _insert_tendencies_to_diagnostics(self, tendencies, diagnostics):
        for name, value in tendencies.items():
            tendency_name = self._get_tendency_name(name)
            diagnostics[tendency_name] = value

    @abc.abstractmethod
    def array_call(self, state, timestep):
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


@add_metaclass(ComponentMeta)
class DiagnosticComponent(object):
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

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    def __str__(self):
        return (
            "instance of {}(DiagnosticComponent)\n"
            "    inputs: {}\n"
            "    diagnostics: {}".format(
                self.__class__,
                self.input_properties.keys(),
                self.diagnostic_properties.keys(),
            )
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

    def __init__(self):
        """
        Initializes the Stepper object.
        """
        self._input_checker = InputChecker(self)
        self._diagnostic_checker = DiagnosticChecker(self)
        self.__initialized = True
        super(DiagnosticComponent, self).__init__()

    def _check_self_is_initialized(self):
        try:
            initialized = self.__initialized
        except AttributeError:
            initialized = False
        if not initialized:
            raise RuntimeError(
                "Component has not called __init__ of base class, likely "
                "because its class {} is missing a call to "
                "super({}, self).__init__(**kwargs) in its __init__ "
                "method.".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )

    def __call__(self, state):
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
        self._check_self_is_initialized()
        self._input_checker.check_inputs(state)
        raw_state = get_arrays_with_properties(state, self.input_properties)
        raw_state["time"] = state["time"]
        raw_diagnostics = self.array_call(raw_state)
        self._diagnostic_checker.check_diagnostics(raw_diagnostics)
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics,
            self.diagnostic_properties,
            state,
            self.input_properties,
        )
        return diagnostics

    @abc.abstractmethod
    def array_call(self, state):
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


@add_metaclass(abc.ABCMeta)
class Monitor(object):
    def __str__(self):
        return "instance of {}(Monitor)".format(self.__class__)

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

    @abc.abstractmethod
    def store(self, state):
        """
        Stores the given state in the Monitor and performs class-specific
        actions.

        Args
        ----
        state: dict
            A model state dictionary.

        Raises
        ------
        InvalidStateError
            If state is not a valid input for the DiagnosticComponent instance.
        """
