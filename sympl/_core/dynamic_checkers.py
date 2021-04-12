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
from typing import Optional, TYPE_CHECKING

from sympl._core.dynamic_operators import (
    InflowComponentOperator,
    OutflowComponentOperator,
)
from sympl._core.static_operators import StaticComponentOperator
from sympl._core.exceptions import (
    InvalidArrayDictError,
    InvalidDataArrayDictError,
    InvalidNDArrayLikeDictError,
)
from sympl._core.factory import AbstractFactory

if TYPE_CHECKING:
    from sympl._core.typingx import (
        Component,
        ArrayDict,
        DataArrayDict,
        NDArrayLikeDict,
    )


class DynamicComponentChecker(abc.ABC):
    properties_name: str = None

    def __init__(self, component: "Component") -> None:
        self.component_name = component.__class__.__name__
        self.properties = getattr(component, self.properties_name, {})
        self.input_properties = getattr(component, "input_properties", {})
        self.static_operator = StaticComponentOperator.factory(
            self.properties_name
        )
        self.aliases = self.static_operator.get_aliases(component)

    def check_missing_fields(self, array_dict: "ArrayDict"):
        """Check if ``array_dict`` contains all keys of ``self.properties``."""
        for name in self.properties:
            if (
                name not in array_dict
                and self.aliases.get(name) not in array_dict
            ):
                raise InvalidArrayDictError(
                    f"Quantity {name} declared in {self.properties_name} of "
                    f"{self.component_name} is missing."
                )


class InflowComponentChecker(DynamicComponentChecker, AbstractFactory):
    def __init__(self, component: "Component") -> None:
        super().__init__(component)
        self.dynamic_operator = InflowComponentOperator.factory(
            self.properties_name, component
        )

    def check_actual_dims(
        self,
        dataarray_dict: "DataArrayDict",
        input_dataarray_dict: Optional["DataArrayDict"] = None,
    ) -> None:
        """Check the dimensions of the DataArrays in ``dataarray_dict``."""
        actual_dims = self.dynamic_operator.get_actual_dims(dataarray_dict)
        target_dims = self.dynamic_operator.get_target_dims(
            dataarray_dict, input_dataarray_dict
        )
        for name in self.properties:
            if any(
                actual_dim not in target_dims[name]
                for actual_dim in actual_dims[name]
            ) or any(
                target_dim not in actual_dims[name]
                for target_dim in target_dims[name]
            ):
                raise InvalidDataArrayDictError(
                    f"According to {self.properties_name} of "
                    f"{self.component_name}, {name} should have dimensions "
                    f"({', '.join(target_dims[name])}) but actually has "
                    f"({', '.join(actual_dims[name])})."
                )

    def check_dim_lengths(self, dataarray_dict: "DataArrayDict") -> None:
        """Check the shape of the DataArrays in ``dataarray_dict``."""
        dim_lengths = self.dynamic_operator.get_dim_lengths(dataarray_dict)
        for dim, lengths in dim_lengths.items():
            if len(lengths) != 1:
                raise InvalidDataArrayDictError(
                    f"{self.component_name}: Dimension {dim} has multiple "
                    f"lengths ({', '.join(lengths)})."
                )

    def check(
        self,
        dataarray_dict: "DataArrayDict",
        input_dataarray_dict: Optional["DataArrayDict"] = None,
    ) -> None:
        """Run all checks on ``dataarray_dict``."""
        self.check_missing_fields(dataarray_dict)
        self.check_actual_dims(dataarray_dict, input_dataarray_dict)
        self.check_dim_lengths(dataarray_dict)


class InputDynamicsComponentChecker(InflowComponentChecker):
    name = "input_properties"
    properties_name = "input_properties"


class OutflowComponentChecker(DynamicComponentChecker, AbstractFactory):
    def __init__(self, component: "Component") -> None:
        super().__init__(component)
        self.dynamic_operator = OutflowComponentOperator.factory(
            self.properties_name, component
        )

    def check_extra_fields(self, ndarray_dict: "NDArrayLikeDict") -> None:
        """
        Check if ``ndarray_dict`` contains any key not present in
        ``self.properties``.
        """
        for name in ndarray_dict:
            if (
                name not in self.properties
                and name not in self.aliases.values()
            ):
                raise InvalidNDArrayLikeDictError(
                    f"{self.component_name} computes {name} which is not "
                    f"declared in {self.properties_name}."
                )

        if len(ndarray_dict) > len(self.properties):
            raise InvalidNDArrayLikeDictError(
                f"{self.component_name} expects an ndarray dict of length "
                f"{len(self.properties)}, but got {len(ndarray_dict)}."
            )

    def check_shape(
        self,
        ndarray_dict: "NDArrayLikeDict",
        input_dataarray_dict: "DataArrayDict",
    ) -> None:
        """Check the shape of the arrays in ``ndarray_dict``."""
        dim_lengths = self.dynamic_operator.get_dim_lengths(
            input_dataarray_dict
        )
        target_dims = self.dynamic_operator.get_target_dims(
            input_dataarray_dict
        )

        for name in self.properties:
            ndarray = self.dynamic_operator.get_field(name, ndarray_dict)
            if len(ndarray.shape) != len(target_dims[name]):
                raise InvalidNDArrayLikeDictError(
                    f"The array for {name} output by {self.component_name} "
                    f"should be {len(target_dims[name])}-dimensional but "
                    f"it is {len(ndarray.shape)}-dimensional."
                )
            for idx, dim in enumerate(target_dims.keys()):
                if dim not in dim_lengths:
                    dim_lengths[dim] = ndarray.shape[idx]
                elif dim_lengths[dim] != ndarray.shape[idx]:
                    raise InvalidNDArrayLikeDictError(
                        f"The array for {name} output by {self.component_name} "
                        f"should have shape {dim_lengths[dim]} along dimension "
                        f"{dim} but it has shape {ndarray.shape[idx]}."
                    )

    def check(
        self,
        ndarray_dict: "NDArrayLikeDict",
        input_dataarray_dict: "DataArrayDict",
    ) -> None:
        """Run all checks on ``ndarray_dict``."""
        self.check_missing_fields(ndarray_dict)
        self.check_extra_fields(ndarray_dict)
        self.check_shape(ndarray_dict, input_dataarray_dict)


class DiagnosticDynamicComponentChecker(OutflowComponentChecker):
    name = "diagnostic_properties"
    properties_name = "diagnostic_properties"


class OutputDynamicComponentChecker(OutflowComponentChecker):
    name = "output_properties"
    properties_name = "output_properties"


class TendencyDynamicComponentChecker(OutflowComponentChecker):
    name = "tendency_properties"
    properties_name = "tendency_properties"
