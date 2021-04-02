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
from typing import Dict, List, Sequence, TYPE_CHECKING, Tuple, Union

from sympl._core.exceptions import (
    ComponentExtraOutputError,
    ComponentMissingOutputError,
    InvalidPropertyDictError,
    InvalidStateError,
    SharedKeyError,
)
from sympl._core.units import (
    get_name_with_incompatible_units,
    get_tendency_name_with_incompatible_units,
)

if TYPE_CHECKING:
    from sympl._core.typing import (
        Component,
        DataArray,
        DataArrayDict,
        NDArrayLike,
        NDArrayLikeDict,
        Property,
        PropertyDict,
    )

    FieldDict = Union[DataArrayDict, NDArrayLikeDict]


class DiagnosticChecker:
    def __init__(self, component: "Component") -> None:
        if not hasattr(component, "diagnostic_properties"):
            raise InvalidPropertyDictError(
                f"Component of type {type(component)} is missing "
                f"diagnostic_properties."
            )
        elif not isinstance(component.diagnostic_properties, dict):
            raise InvalidPropertyDictError(
                f"diagnostic_properties on component of type {type(component)} "
                f"is of type {type(component.input_properties)}, but "
                f"should be an instance of dict."
            )

        for name, properties in component.diagnostic_properties.items():
            if "units" not in properties:
                raise InvalidPropertyDictError(
                    f"DiagnosticComponent properties do not have units defined "
                    f"for {name}."
                )
            if (
                "dims" not in properties
                and name not in component.input_properties
            ):
                raise InvalidPropertyDictError(
                    f"DiagnosticComponent properties do not have dims defined "
                    f"for {name}."
                )

        incompatible_name = get_name_with_incompatible_units(
            component.input_properties, component.diagnostic_properties,
        )
        if incompatible_name is not None:
            raise InvalidPropertyDictError(
                f"Component of type {type(component)} has input "
                f"{incompatible_name} with diagnostic units "
                f"{component.diagnostic_properties[incompatible_name]['units']} "
                f"that are incompatible with input units "
                f"{component.input_properties[incompatible_name]['units']}."
            )

        check_overlapping_aliases(
            component.diagnostic_properties, "diagnostic"
        )

        self.component = component
        self._ignored_diagnostics = []

    def _get_diagnostic_aliases(self) -> Dict[str, List[str]]:
        wanted_diagnostic_aliases = {}
        for name, properties in self.component.diagnostic_properties.items():
            wanted_diagnostic_aliases[name] = []
            if "alias" in properties:
                wanted_diagnostic_aliases[name].append(properties["alias"])
            if (
                name in self.component.input_properties
                and "alias" in self.component.input_properties[name]
            ):
                wanted_diagnostic_aliases[name].append(
                    self.component.input_properties[name]["alias"]
                )
        return wanted_diagnostic_aliases

    def _check_missing_diagnostics(self, diagnostic_dict: "FieldDict") -> None:
        missing_diagnostics = set()
        diagnostic_aliases = self._get_diagnostic_aliases()

        for name, aliases in diagnostic_aliases.items():
            if (
                name not in diagnostic_dict
                and name not in self._ignored_diagnostics
                and not any(alias in diagnostic_dict for alias in aliases)
            ):
                missing_diagnostics.add(name)

        if len(missing_diagnostics) > 0:
            raise ComponentMissingOutputError(
                f"Component {type(self.component)} did not compute "
                f"diagnostic(s) {', '.join(missing_diagnostics)}."
            )

    def _check_extra_diagnostics(self, diagnostic_dict: "FieldDict") -> None:
        diagnostic_aliases = self._get_diagnostic_aliases()

        wanted_set = set()
        wanted_set.update(diagnostic_aliases.keys())
        for value_list in diagnostic_aliases.values():
            wanted_set.update(value_list)

        extra_diagnostics = set(diagnostic_dict.keys()).difference(wanted_set)
        if len(extra_diagnostics) > 0:
            raise ComponentExtraOutputError(
                f"Component {type(self.component)} computed diagnostic(s) "
                f"{', '.join(extra_diagnostics)} which are not in "
                f"diagnostic_properties."
            )

    @property
    def ignored_diagnostics(self) -> Tuple[str]:
        return tuple(self._ignored_diagnostics)

    @ignored_diagnostics.setter
    def ignored_diagnostics(self, val: List[str]) -> None:
        self._ignored_diagnostics = val

    def check_diagnostics(self, diagnostics_dict: "FieldDict") -> None:
        self._check_missing_diagnostics(diagnostics_dict)
        self._check_extra_diagnostics(diagnostics_dict)


class InputChecker:
    def __init__(self, component: "Component") -> None:
        if not hasattr(component, "input_properties"):
            raise InvalidPropertyDictError(
                f"Component of type {type(component)} is missing "
                f"input_properties."
            )
        if not isinstance(component.input_properties, dict):
            raise InvalidPropertyDictError(
                f"input_properties on component of type {type(component)} "
                f"is of type {type(component.input_properties)}, "
                f"but should be an instance of dict."
            )

        for name, properties in component.input_properties.items():
            if "units" not in properties:
                raise InvalidPropertyDictError(
                    f"Input properties do not have units defined for {name}."
                )
            if "dims" not in properties:
                raise InvalidPropertyDictError(
                    f"Input properties do not have dims defined for {name}."
                )

        check_overlapping_aliases(component.input_properties, "input")

        self.component = component

    def check_inputs(self, state: "FieldDict") -> None:
        for name in self.component.input_properties:
            if name not in state:
                raise InvalidStateError(f"Missing input quantity {name}.")


class OutputChecker:
    def __init__(self, component: "Component") -> None:
        if not hasattr(component, "output_properties"):
            raise InvalidPropertyDictError(
                f"Component of type {type(component)} is missing "
                f"output_properties."
            )
        elif not isinstance(component.output_properties, dict):
            raise InvalidPropertyDictError(
                f"output_properties on component of type {type(component)} is "
                f"of type {type(component(component.output_properties))}, but "
                f"should be an instance of dict."
            )

        for name, properties in component.output_properties.items():
            if "units" not in properties:
                raise InvalidPropertyDictError(
                    f"Output properties do not have units defined for {name}."
                )
            if (
                "dims" not in properties
                and name not in component.input_properties
            ):
                raise InvalidPropertyDictError(
                    f"Output properties do not have dims defined for {name}."
                )

        check_overlapping_aliases(component.output_properties, "output")

        incompatible_name = get_name_with_incompatible_units(
            component.input_properties, component.output_properties
        )
        if incompatible_name is not None:
            raise InvalidPropertyDictError(
                f"Component of type {type(component)} has input "
                f"{incompatible_name} with output units "
                f"{component.output_properties[incompatible_name]['units']} "
                f"that are incompatible with input units "
                f"{component.input_properties[incompatible_name]['units']}."
            )

        self.component = component

    def _get_output_aliases(self) -> Dict[str, List[str]]:
        wanted_output_aliases = {}
        for name, properties in self.component.output_properties.items():
            wanted_output_aliases[name] = []
            if "alias" in properties:
                wanted_output_aliases[name].append(properties["alias"])
            if (
                name in self.component.input_properties
                and "alias" in self.component.input_properties[name]
            ):
                wanted_output_aliases[name].append(
                    self.component.input_properties[name]["alias"]
                )
        return wanted_output_aliases

    def _check_missing_outputs(self, output_dict: "FieldDict") -> None:
        output_aliases = self._get_output_aliases()

        missing_outputs = set()
        for name, aliases in output_aliases.items():
            if name not in output_dict and not any(
                alias in output_dict for alias in aliases
            ):
                missing_outputs.add(name)

        if len(missing_outputs) > 0:
            raise ComponentMissingOutputError(
                f"Component {type(self.component)} did not compute output(s) "
                f"{', '.join(missing_outputs)}."
            )

    def _check_extra_outputs(self, output_dict: "FieldDict") -> None:
        output_aliases = self._get_output_aliases()

        wanted_set = set()
        wanted_set.update(output_aliases.keys())
        for value_list in output_aliases.values():
            wanted_set.update(value_list)

        extra_outputs = set(output_dict.keys()).difference(wanted_set)
        if len(extra_outputs) > 0:
            raise ComponentExtraOutputError(
                f"Component {type(self.component)} computed output(s) "
                f"{', '.join(extra_outputs)} which are not in "
                f"output_properties."
            )

    def check_outputs(self, output_dict: "FieldDict") -> None:
        self._check_missing_outputs(output_dict)
        self._check_extra_outputs(output_dict)


class TendencyChecker:
    def __init__(self, component: "Component") -> None:
        if not hasattr(component, "tendency_properties"):
            raise InvalidPropertyDictError(
                f"Component of type {type(component)} is missing "
                f"tendency_properties."
            )
        if not isinstance(component.tendency_properties, dict):
            raise InvalidPropertyDictError(
                f"tendency_properties on component of type {type(component)} "
                f"is of type {type(component.input_properties)}, "
                f"but should be an instance of dict."
            )

        for name, properties in component.tendency_properties.items():
            if "units" not in properties:
                raise InvalidPropertyDictError(
                    f"Tendency properties do not have units defined for {name}."
                )
            if (
                "dims" not in properties
                and name not in component.input_properties
            ):
                raise InvalidPropertyDictError(
                    f"Tendency properties do not have dims defined for {name}."
                )

        check_overlapping_aliases(component.tendency_properties, "tendency")
        incompatible_name = get_tendency_name_with_incompatible_units(
            component.input_properties, component.tendency_properties
        )
        if incompatible_name is not None:
            raise InvalidPropertyDictError(
                f"Component of type {type(component)} has input "
                f"{incompatible_name} with tendency units "
                f"{component.tendency_properties[incompatible_name]['units']} "
                f"that are incompatible with input units "
                f"{component.input_properties[incompatible_name]['units']}"
            )

        self.component = component

    def _get_tendency_aliases(self) -> Dict[str, List[str]]:
        wanted_tendency_aliases = {}
        for name, properties in self.component.tendency_properties.items():
            wanted_tendency_aliases[name] = []
            if "alias" in properties:
                wanted_tendency_aliases[name].append(properties["alias"])
            if (
                name in self.component.input_properties
                and "alias" in self.component.input_properties[name]
            ):
                wanted_tendency_aliases[name].append(
                    self.component.input_properties[name]["alias"]
                )
        return wanted_tendency_aliases

    def _check_missing_tendencies(self, tendency_dict: "FieldDict") -> None:
        missing_tendencies = set()
        tendency_aliases = self._get_tendency_aliases()

        for name, aliases in tendency_aliases.items():
            if name not in tendency_dict and not any(
                alias in tendency_dict for alias in aliases
            ):
                missing_tendencies.add(name)

        if len(missing_tendencies) > 0:
            raise ComponentMissingOutputError(
                f"Component {type(self.component)} did not compute tendencies "
                f"for {', '.join(missing_tendencies)}."
            )

    def _check_extra_tendencies(self, tendency_dict: "FieldDict") -> None:
        tendency_aliases = self._get_tendency_aliases()

        wanted_set = set()
        wanted_set.update(tendency_aliases.keys())
        for value_list in tendency_aliases.values():
            wanted_set.update(value_list)

        extra_tendencies = set(tendency_dict.keys()).difference(wanted_set)
        if len(extra_tendencies) > 0:
            raise ComponentExtraOutputError(
                f"Component {type(self.component)} computed tendencies for "
                f"{', '.join(extra_tendencies)} which are not in "
                f"tendency_properties."
            )

    def check_tendencies(self, tendency_dict: "FieldDict") -> None:
        self._check_missing_tendencies(tendency_dict)
        self._check_extra_tendencies(tendency_dict)


def check_array_shape(
    out_dims: Sequence[str],
    array: "NDArrayLike",
    name: str,
    dim_lengths: Dict[str, int],
) -> None:
    if len(out_dims) != len(array.shape):
        raise InvalidPropertyDictError(
            f"Returned array for {name} has shape {array.shape} "
            f"which is incompatible with dims {out_dims} in properties."
        )
    for dim, length in zip(out_dims, array.shape):
        if dim in dim_lengths and dim_lengths[dim] != length:
            raise InvalidPropertyDictError(
                f"Dimension {dim} of quantity {name} has length {length}, "
                f"but another quantity has length {dim_lengths[dim]}."
            )


def check_overlapping_aliases(
    properties: "PropertyDict", properties_name: str
) -> None:
    defined_aliases = set()
    for name, properties in properties.items():
        if "alias" in properties:
            if properties["alias"] not in defined_aliases:
                defined_aliases.add(properties["alias"])
            else:
                raise InvalidPropertyDictError(
                    f"Multiple quantities map to alias {properties['alias']} "
                    f"in {properties_name} properties."
                )


def ensure_no_shared_keys(dict1: Dict, dict2: Dict) -> None:
    """
    Raises SharedKeyError if there exists a key present in both
    dictionaries.
    """
    shared_keys = [key for key in dict1 if key in dict2]
    if len(shared_keys) > 0:
        raise SharedKeyError(
            f"Unexpected shared keys: {', '.join(shared_keys)}."
        )


def ensure_properties_have_dims_and_units(
    properties: "Property", quantity_name: str
) -> None:
    if "dims" not in properties:
        raise InvalidPropertyDictError(
            f"dims not specified for quantity {quantity_name}."
        )
    if "units" not in properties:
        raise InvalidPropertyDictError(
            f"units not specified for quantity {quantity_name}."
        )


def ensure_quantity_has_units(
    quantity: "DataArray", quantity_name: str
) -> None:
    if "units" not in quantity.attrs:
        raise InvalidStateError(
            f"Quantity {quantity_name} is missing units attribute."
        )


def ensure_values_are_arrays(array_dict):
    pass
    # for name, value in array_dict.items():
    # if not isinstance(value, np.ndarray):
    # array_dict[name] = np.asarray(value)
