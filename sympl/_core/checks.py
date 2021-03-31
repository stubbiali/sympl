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
from sympl._core.exceptions import (
    InvalidPropertyDictError,
    InvalidStateError,
    ComponentMissingOutputError,
    ComponentExtraOutputError,
)
from sympl._core.units import (
    get_name_with_incompatible_units,
    get_tendency_name_with_incompatible_units,
)


class InputChecker(object):
    def __init__(self, component):
        self.component = component
        if not hasattr(component, "input_properties"):
            raise InvalidPropertyDictError(
                "Component of type {} is missing input_properties".format(
                    component.__class__.__name__
                )
            )
        elif not isinstance(component.input_properties, dict):
            raise InvalidPropertyDictError(
                "input_properties on component of type {} is of type {}, but "
                "should be an instance of dict".format(
                    component.__class__.__name__,
                    component.input_properties.__class__,
                )
            )
        for name, properties in self.component.input_properties.items():
            if "units" not in properties.keys():
                raise InvalidPropertyDictError(
                    "Input properties do not have units defined for {}".format(
                        name
                    )
                )
            if "dims" not in properties.keys():
                raise InvalidPropertyDictError(
                    "Input properties do not have dims defined for {}".format(
                        name
                    )
                )
        check_overlapping_aliases(self.component.input_properties, "input")
        super(InputChecker, self).__init__()

    def check_inputs(self, state):
        for key in self.component.input_properties.keys():
            if key not in state.keys():
                raise InvalidStateError(
                    "Missing input quantity {}".format(key)
                )


class TendencyChecker(object):
    def __init__(self, component):
        self.component = component
        if not hasattr(component, "tendency_properties"):
            raise InvalidPropertyDictError(
                "Component of type {} is missing tendency_properties".format(
                    component.__class__.__name__
                )
            )
        elif not isinstance(component.tendency_properties, dict):
            raise InvalidPropertyDictError(
                "tendency_properties on component of type {} is of type {}, but "
                "should be an instance of dict".format(
                    component.__class__.__name__,
                    component.input_properties.__class__,
                )
            )
        for name, properties in self.component.tendency_properties.items():
            if "units" not in properties.keys():
                raise InvalidPropertyDictError(
                    "Tendency properties do not have units defined for {}".format(
                        name
                    )
                )
            if (
                "dims" not in properties.keys()
                and name not in self.component.input_properties.keys()
            ):
                raise InvalidPropertyDictError(
                    "Tendency properties do not have dims defined for {}".format(
                        name
                    )
                )
        check_overlapping_aliases(
            self.component.tendency_properties, "tendency"
        )
        incompatible_name = get_tendency_name_with_incompatible_units(
            self.component.input_properties, self.component.tendency_properties
        )
        if incompatible_name is not None:
            raise InvalidPropertyDictError(
                "Component of type {} has input {} with tendency units {} that "
                "are incompatible with input units {}".format(
                    type(self.component),
                    incompatible_name,
                    self.component.tendency_properties[incompatible_name][
                        "units"
                    ],
                    self.component.input_properties[incompatible_name][
                        "units"
                    ],
                )
            )
        super(TendencyChecker, self).__init__()

    @property
    def _wanted_tendency_aliases(self):
        wanted_tendency_aliases = {}
        for name, properties in self.component.tendency_properties.items():
            wanted_tendency_aliases[name] = []
            if "alias" in properties.keys():
                wanted_tendency_aliases[name].append(properties["alias"])
            if (
                name in self.component.input_properties.keys()
                and "alias" in self.component.input_properties[name].keys()
            ):
                wanted_tendency_aliases[name].append(
                    self.component.input_properties[name]["alias"]
                )
        return wanted_tendency_aliases

    def _check_missing_tendencies(self, tendency_dict):
        missing_tendencies = set()
        for name, aliases in self._wanted_tendency_aliases.items():
            if name not in tendency_dict.keys() and not any(
                alias in tendency_dict.keys() for alias in aliases
            ):
                missing_tendencies.add(name)
        if len(missing_tendencies) > 0:
            raise ComponentMissingOutputError(
                "Component {} did not compute tendencies for {}".format(
                    self.component.__class__.__name__,
                    ", ".join(missing_tendencies),
                )
            )

    def _check_extra_tendencies(self, tendency_dict):
        wanted_set = set()
        wanted_set.update(self._wanted_tendency_aliases.keys())
        for value_list in self._wanted_tendency_aliases.values():
            wanted_set.update(value_list)
        extra_tendencies = set(tendency_dict.keys()).difference(wanted_set)
        if len(extra_tendencies) > 0:
            raise ComponentExtraOutputError(
                "Component {} computed tendencies for {} which are not in "
                "tendency_properties".format(
                    self.component.__class__.__name__,
                    ", ".join(extra_tendencies),
                )
            )

    def check_tendencies(self, tendency_dict):
        self._check_missing_tendencies(tendency_dict)
        self._check_extra_tendencies(tendency_dict)


class DiagnosticChecker(object):
    def __init__(self, component):
        self.component = component
        if not hasattr(component, "diagnostic_properties"):
            raise InvalidPropertyDictError(
                "Component of type {} is missing diagnostic_properties".format(
                    component.__class__.__name__
                )
            )
        elif not isinstance(component.diagnostic_properties, dict):
            raise InvalidPropertyDictError(
                "diagnostic_properties on component of type {} is of type {}, but "
                "should be an instance of dict".format(
                    component.__class__.__name__,
                    component.input_properties.__class__,
                )
            )
        self._ignored_diagnostics = []
        for name, properties in component.diagnostic_properties.items():
            if "units" not in properties.keys():
                raise InvalidPropertyDictError(
                    "DiagnosticComponent properties do not have units defined for {}".format(
                        name
                    )
                )
            if (
                "dims" not in properties.keys()
                and name not in component.input_properties.keys()
            ):
                raise InvalidPropertyDictError(
                    "DiagnosticComponent properties do not have dims defined for {}".format(
                        name
                    )
                )
        incompatible_name = get_name_with_incompatible_units(
            self.component.input_properties,
            self.component.diagnostic_properties,
        )
        if incompatible_name is not None:
            raise InvalidPropertyDictError(
                "Component of type {} has input {} with diagnostic units {} that "
                "are incompatible with input units {}".format(
                    type(self.component),
                    incompatible_name,
                    self.component.diagnostic_properties[incompatible_name][
                        "units"
                    ],
                    self.component.input_properties[incompatible_name][
                        "units"
                    ],
                )
            )
        check_overlapping_aliases(
            component.diagnostic_properties, "diagnostic"
        )

    @property
    def _wanted_diagnostic_aliases(self):
        wanted_diagnostic_aliases = {}
        for name, properties in self.component.diagnostic_properties.items():
            wanted_diagnostic_aliases[name] = []
            if "alias" in properties.keys():
                wanted_diagnostic_aliases[name].append(properties["alias"])
            if (
                name in self.component.input_properties.keys()
                and "alias" in self.component.input_properties[name].keys()
            ):
                wanted_diagnostic_aliases[name].append(
                    self.component.input_properties[name]["alias"]
                )
        return wanted_diagnostic_aliases

    def _check_missing_diagnostics(self, diagnostics_dict):
        missing_diagnostics = set()
        for name, aliases in self._wanted_diagnostic_aliases.items():
            if (
                name not in diagnostics_dict.keys()
                and name not in self._ignored_diagnostics
                and not any(
                    alias in diagnostics_dict.keys() for alias in aliases
                )
            ):
                missing_diagnostics.add(name)
        if len(missing_diagnostics) > 0:
            raise ComponentMissingOutputError(
                "Component {} did not compute diagnostic(s) {}".format(
                    self.component.__class__.__name__,
                    ", ".join(missing_diagnostics),
                )
            )

    def _check_extra_diagnostics(self, diagnostics_dict):
        wanted_set = set()
        wanted_set.update(self._wanted_diagnostic_aliases.keys())
        for value_list in self._wanted_diagnostic_aliases.values():
            wanted_set.update(value_list)
        extra_diagnostics = set(diagnostics_dict.keys()).difference(wanted_set)
        if len(extra_diagnostics) > 0:
            raise ComponentExtraOutputError(
                "Component {} computed diagnostic(s) {} which are not in "
                "diagnostic_properties".format(
                    self.component.__class__.__name__,
                    ", ".join(extra_diagnostics),
                )
            )

    def set_ignored_diagnostics(self, ignored_diagnostics):
        self._ignored_diagnostics = ignored_diagnostics

    def check_diagnostics(self, diagnostics_dict):
        self._check_missing_diagnostics(diagnostics_dict)
        self._check_extra_diagnostics(diagnostics_dict)


class OutputChecker(object):
    def __init__(self, component):
        self.component = component
        if not hasattr(component, "output_properties"):
            raise InvalidPropertyDictError(
                "Component of type {} is missing output_properties".format(
                    component.__class__.__name__
                )
            )
        elif not isinstance(component.output_properties, dict):
            raise InvalidPropertyDictError(
                "output_properties on component of type {} is of type {}, but "
                "should be an instance of dict".format(
                    component.__class__.__name__,
                    component.input_properties.__class__,
                )
            )
        for name, properties in self.component.output_properties.items():
            if "units" not in properties.keys():
                raise InvalidPropertyDictError(
                    "Output properties do not have units defined for {}".format(
                        name
                    )
                )
            if (
                "dims" not in properties.keys()
                and name not in self.component.input_properties.keys()
            ):
                raise InvalidPropertyDictError(
                    "Output properties do not have dims defined for {}".format(
                        name
                    )
                )
        check_overlapping_aliases(self.component.output_properties, "output")
        incompatible_name = get_name_with_incompatible_units(
            self.component.input_properties, self.component.output_properties
        )
        if incompatible_name is not None:
            raise InvalidPropertyDictError(
                "Component of type {} has input {} with output units {} that "
                "are incompatible with input units {}".format(
                    type(self.component),
                    incompatible_name,
                    self.component.output_properties[incompatible_name][
                        "units"
                    ],
                    self.component.input_properties[incompatible_name][
                        "units"
                    ],
                )
            )
        super(OutputChecker, self).__init__()

    @property
    def _wanted_output_aliases(self):
        wanted_output_aliases = {}
        for name, properties in self.component.output_properties.items():
            wanted_output_aliases[name] = []
            if "alias" in properties.keys():
                wanted_output_aliases[name].append(properties["alias"])
            if (
                name in self.component.input_properties.keys()
                and "alias" in self.component.input_properties[name].keys()
            ):
                wanted_output_aliases[name].append(
                    self.component.input_properties[name]["alias"]
                )
        return wanted_output_aliases

    def _check_missing_outputs(self, outputs_dict):
        missing_outputs = set()
        for name, aliases in self._wanted_output_aliases.items():
            if name not in outputs_dict.keys() and not any(
                alias in outputs_dict.keys() for alias in aliases
            ):
                missing_outputs.add(name)
        if len(missing_outputs) > 0:
            raise ComponentMissingOutputError(
                "Component {} did not compute output(s) {}".format(
                    self.component.__class__.__name__,
                    ", ".join(missing_outputs),
                )
            )

    def _check_extra_outputs(self, outputs_dict):
        wanted_set = set()
        wanted_set.update(self._wanted_output_aliases.keys())
        for value_list in self._wanted_output_aliases.values():
            wanted_set.update(value_list)
        extra_outputs = set(outputs_dict.keys()).difference(wanted_set)
        if len(extra_outputs) > 0:
            raise ComponentExtraOutputError(
                "Component {} computed output(s) {} which are not in "
                "output_properties".format(
                    self.component.__class__.__name__, ", ".join(extra_outputs)
                )
            )

    def check_outputs(self, output_dict):
        self._check_missing_outputs(output_dict)
        self._check_extra_outputs(output_dict)


def check_overlapping_aliases(properties, properties_name):
    defined_aliases = set()
    for name, properties in properties.items():
        if "alias" in properties.keys():
            if properties["alias"] not in defined_aliases:
                defined_aliases.add(properties["alias"])
            else:
                raise InvalidPropertyDictError(
                    "Multiple quantities map to alias {} in {} "
                    "properties".format(properties["alias"], properties_name)
                )


def ensure_values_are_arrays(array_dict):
    pass
    # for name, value in array_dict.items():
    # if not isinstance(value, np.ndarray):
    # array_dict[name] = np.asarray(value)


def check_array_shape(out_dims, raw_array, name, dim_lengths):
    if len(out_dims) != len(raw_array.shape):
        raise InvalidPropertyDictError(
            "Returned array for {} has shape {} "
            "which is incompatible with dims {} in properties".format(
                name, raw_array.shape, out_dims
            )
        )
    for dim, length in zip(out_dims, raw_array.shape):
        if dim in dim_lengths.keys() and dim_lengths[dim] != length:
            raise InvalidPropertyDictError(
                "Dimension {} of quantity {} has length {}, but "
                "another quantity has length {}".format(
                    dim, name, length, dim_lengths[dim]
                )
            )
