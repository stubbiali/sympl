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
from typing import Callable, Dict, Optional, Sequence, TYPE_CHECKING

from sympl._core.factory import AbstractFactory

if TYPE_CHECKING:
    from sympl._core.typingx import Component, PropertyDict


class StaticComponentOperator(abc.ABC, AbstractFactory):
    properties_name: str = None

    @classmethod
    def get_aliases(cls, component: "Component") -> Dict[str, str]:
        """Get the alias for each field."""
        properties = getattr(component, cls.properties_name, {})
        input_properties = getattr(component, "input_properties", {})

        out = {}
        for name in properties:
            if "alias" in properties[name]:
                out[name] = properties[name]["alias"]
            elif (
                name in input_properties and "alias" in input_properties[name]
            ):
                out[name] = input_properties[name]["alias"]

        return out

    @classmethod
    @abc.abstractmethod
    def get_allocator(cls, component: "Component") -> Optional[Callable]:
        pass

    @classmethod
    def _get_dims(
        cls, component: "Component", other_properties_name: str,
    ) -> Dict[str, Sequence[str]]:
        """Get the specification of the dimensions for each field."""
        properties = cls.get_properties(component)
        other_properties = StaticComponentOperator.factory(
            other_properties_name
        ).get_properties(component)

        out = {}
        for name in properties:
            if "dims" in properties[name]:
                out[name] = properties[name]["dims"]
            elif "dims_like" in properties[name]:
                field_like = properties[name]["dims_like"]
                out[name] = (
                    properties[field_like]["dims"]
                    if field_like in properties
                    else other_properties[field_like]["dims"]
                )

        return out

    @classmethod
    def get_dims(cls, component: "Component",) -> Dict[str, Sequence[str]]:
        return cls._get_dims(component, "input_properties")

    @classmethod
    def get_properties(cls, component: "Component") -> "PropertyDict":
        return getattr(component, cls.properties_name, {})

    @classmethod
    def get_properties_with_dims(
        cls, component: "Component"
    ) -> "PropertyDict":
        out = cls.get_properties(component).copy()
        dims = cls.get_dims(component)
        for name in dims:
            if name in out:
                out[name]["dims"] = dims[name]
        return out


class DiagnosticStaticComponentOperator(StaticComponentOperator):
    name = "diagnostic_properties"
    properties_name = "diagnostic_properties"

    @classmethod
    def get_allocator(cls, component: "Component") -> Optional[Callable]:
        return getattr(component, "allocate_diagnostic", None)


class InputStaticComponentOperator(StaticComponentOperator):
    name = "input_properties"
    properties_name = "input_properties"

    @classmethod
    def get_allocator(cls, component: "Component") -> Optional[Callable]:
        return None


class OutputStaticComponentOperator(StaticComponentOperator):
    name = "output_properties"
    properties_name = "output_properties"

    @classmethod
    def get_allocator(cls, component: "Component") -> Optional[Callable]:
        return getattr(component, "allocate_output", None)


class TendencyStaticComponentOperator(StaticComponentOperator):
    name = "tendency_properties"
    properties_name = "tendency_properties"

    @classmethod
    def get_allocator(cls, component: "Component") -> Optional[Callable]:
        return getattr(component, "allocate_tendency", None)
