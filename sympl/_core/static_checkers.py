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
from typing import TYPE_CHECKING

from sympl._core.exceptions import InvalidPropertyDictError
from sympl._core.factory import AbstractFactory
from sympl._core.static_operators import StaticComponentOperator
from sympl._core.units import units_are_compatible

if TYPE_CHECKING:
    from sympl._core.typingx import Component, PropertyDict


class StaticComponentChecker(abc.ABC, AbstractFactory):
    properties_name: str = None

    @classmethod
    def check_component_is_initialized(cls, component: "Component") -> None:
        """
        Check if ``component`` has the attribute ``_initialized`` and if
        this is ``True``.
        """
        init = getattr(component, "_initialized", False)
        if not init:
            name = component.__class__.__name__
            raise RuntimeError(
                f"The __init__ method of {name} is missing a call to "
                f"super({name}, self).__init__(**kwargs)."
            )

    @classmethod
    def check_component_has_property(cls, component: "Component") -> None:
        """
        Check if ``component`` has an attribute named  ``cls.properties_name``.
        """
        if not hasattr(component, cls.properties_name):
            raise InvalidPropertyDictError(
                f"{component.__class__.__name__} misses property "
                f"{cls.properties_name}."
            )

    @classmethod
    def check_property_type(
        cls, component: "Component", properties: "PropertyDict"
    ) -> None:
        """
        Check if ``properties`` - the attribute ``cls.properties_name`` of
        ``component`` - is a dict.
        """
        if not isinstance(properties, dict):
            raise InvalidPropertyDictError(
                f"{cls.properties_name} on component "
                f"{component.__class__.__name__} is of type "
                f"{properties.__class__.__name__}, "
                f"but should be an instance of dict."
            )

    @classmethod
    def check_properties_have_units(
        cls,
        component: "Component",
        properties: "PropertyDict",
        other_properties: "PropertyDict",
    ) -> None:
        """
        Check if each key-value pair of ``properties`` satisfies one of the
        following conditions:

            * value contains ``"units"``;
            * key is found also in ``other_properties``.
        """
        for field_name, field_properties in properties.items():
            if "units" not in field_properties:
                raise InvalidPropertyDictError(
                    f"{component.__class__.__name__} does not define units for "
                    f"{field_name} in {cls.properties_name}."
                )

            if (
                "dims" not in field_properties
                and field_name not in other_properties
            ):
                raise InvalidPropertyDictError(
                    f"{component.__class__.__name__} does not define dims for "
                    f"{field_name} in {cls.properties_name}."
                )

    @staticmethod
    def wrap_units(units: str) -> str:
        """Modify ``units``."""
        return units

    @classmethod
    def check_incompatible_units(
        cls,
        component: "Component",
        properties: "PropertyDict",
        other_properties_name: str,
        other_properties: "PropertyDict",
    ) -> None:
        """
        If ``name`` is found both in ``properties`` and ``other_properties``,
        check if the units specified in the two dictionaries are compatible.
        """
        for name in properties:
            if name in other_properties and not units_are_compatible(
                properties[name]["units"],
                cls.wrap_units(other_properties[name]["units"]),
            ):
                raise InvalidPropertyDictError(
                    f"{component.__class__.__name__} specifies incompatible "
                    f"units for {name}: {other_properties[name]['units']} in "
                    f"{other_properties_name} and "
                    f"{properties[name]['units']} in {cls.properties_name}."
                )

    @classmethod
    def check_properties_have_dims(
        cls,
        component: "Component",
        properties: "PropertyDict",
        other_properties: "PropertyDict",
    ) -> None:
        """
        Check if each key-value pair of ``properties`` satisfies one of the
        following conditions:

            * value contains either ``"dims"``, ``"dims_like"`` or
                ``"match_dims_like"``;
            * key is found also in ``other_properties``.
        """
        for field_name, field_properties in properties.items():
            if (
                "dims" not in field_properties
                and "dims_like" not in field_properties
                and "match_dims_like" not in field_properties
                and field_name not in other_properties
            ):
                raise InvalidPropertyDictError(
                    f"{component.__class__.__name__} does not define dims for "
                    f"{field_name} in {cls.properties_name}."
                )

    @classmethod
    def check_dims_like(
        cls,
        component: "Component",
        properties: "PropertyDict",
        other_properties: "PropertyDict",
    ) -> None:
        """
        If the value ``val`` associated with the key ``name`` of ``properties``
        contains ``"dims_like"``, check if:

            * `val` does not contain ``"dims"``;
            * ``val["dims_like"]`` is found either in ``properties``
                or ``other_properties``;
            * the value ``val_like`` associated with the key
                ``val["dims_like"]`` contains ``"dims"``;
            * ``val_like["dims"]`` does not contain wildcard dimensions.
        """
        for field_name, field_properties in properties.items():
            if "dims_like" in field_properties:
                if "dims" in field_properties:
                    raise InvalidPropertyDictError(
                        f"{cls.properties_name} of "
                        f"{component.__class__.__name__} defines both dims "
                        f"and dims_like for quantity {field_name}."
                    )

                field_like = field_properties["dims_like"]
                if (
                    field_like not in properties
                    and field_like not in other_properties
                ):
                    raise InvalidPropertyDictError(
                        f"Cannot retrieve dims for {field_like} in "
                        f"{component.__class__.__name__}."
                    )

                properties_like = properties.get(
                    field_like, other_properties.get(field_like)
                )
                if "dims" not in properties_like:
                    raise InvalidPropertyDictError(
                        f"{cls.properties_name} of "
                        f"{component.__class__.__name__} missing dims for "
                        f"{field_like}."
                    )
                if "*" in properties_like["dims"]:
                    raise InvalidPropertyDictError(
                        f"{cls.properties_name} of "
                        f"{component.__class__.__name__} defines wildcard "
                        f"dims for {field_like}."
                    )

    @classmethod
    def check_wildcard_dims(
        cls,
        component: "Component",
        properties: "PropertyDict",
        other_properties: "PropertyDict",
    ) -> None:
        """
        If the field ``"dims"`` of the value ``val`` associated with the key
        ``name`` of ``properties`` contains a wildcard character (*), check if:

            * No more than one wildcard character is present;
            * Either ``val`` contains ``"match_dims_like"`` or ``name`` is
                found also in ``other_properties``.
        """
        for name in properties:
            out = sum([dim == "*" for dim in properties[name].get("dims", [])])
            if out == 1:
                if (
                    "match_dims_like" not in properties[name]
                    and name not in other_properties
                ):
                    raise InvalidPropertyDictError(
                        f"{component.__class__.__name__} cannot determine the "
                        f"actual dimensions for {name} based on "
                        f"{cls.properties_name}."
                    )
            elif out > 1:
                raise InvalidPropertyDictError(
                    f"Found multiple wildcard dimensions ('*') in "
                    f"{cls.properties_name} of {component.__class__.__name__} "
                    f"for quantity {name}."
                )

    @classmethod
    def check_match_dims_like(
        cls,
        component: "Component",
        properties: "PropertyDict",
        other_properties: "PropertyDict",
    ) -> None:
        """
        If the value ``val`` associated with the key ``name`` of ``properties``
        contains ``"match_dims_like"``, check if:

            * ``val`` contains ``"dims"``;
            * ``val["dims"]`` contains one wildcard character;
            * ``val["match_dims_like"]`` is either in ``properties`` or
                ``other_properties``;
            * the value ``val_like`` associated with the key
                ``val["match_dims_like"]`` contains ``"dims"``;
            * ``val_like["dims"]`` contains a wildcard character at the same
                location as ``val["dims"]``.
        """
        for field_name, field_properties in properties.items():
            if "match_dims_like" in field_properties:
                if "dims" not in field_properties:
                    raise InvalidPropertyDictError(
                        f"{cls.properties_name} of "
                        f"{component.__class__.__name__} specifies "
                        f"match_dims_like but not dims for {field_name}."
                    )

                if "*" not in field_properties["dims"]:
                    raise InvalidPropertyDictError(
                        f"{component.__class__.__name__} defines "
                        f"match_dims_like for {field_name} in "
                        f"{cls.properties_name} but no wildcard dimensions "
                        f"for {field_name} are given."
                    )

                dims = field_properties["dims"]
                field_like = field_properties["dims_like"]

                if field_like in properties:
                    dims_like = properties[field_like]["dims"]
                elif field_like in other_properties:
                    dims_like = other_properties[field_like]["dims"]
                else:
                    raise InvalidPropertyDictError(
                        f"Cannot retrieve dims for {field_like} in "
                        f"{component.__class__.__name__}."
                    )

                if "*" not in dims_like:
                    raise InvalidPropertyDictError(
                        f"{component.__class__.__name__} defines "
                        f"match_dims_like for {field_name} in "
                        f"{cls.properties_name} but no wildcard dimensions "
                        f"for {field_like} are given."
                    )
                if dims_like.index("*") != dims.index("*"):
                    raise InvalidPropertyDictError(
                        f"{component.__class__.__name__} defines incompatible "
                        f"wildcard dimensions for {field_name} and "
                        f"{field_like}."
                    )

    @classmethod
    def check_aliases(
        cls,
        component: "Component",
        properties: "PropertyDict",
        other_properties_name: str,
        other_properties: "PropertyDict",
    ) -> None:
        """
        Two field names cannot map to the same alias, and any field name
        cannot have more than one alias.
        """
        name_to_alias = {}
        alias_to_name = {}

        for name in other_properties:
            if "alias" in other_properties[name]:
                alias = other_properties[name]["alias"]
                name_to_alias[name] = alias
                alias_to_name[alias] = name

        for name in properties:
            if "alias" in properties[name]:
                new_alias = name_to_alias.setdefault(
                    name, properties[name]["alias"]
                )
                if name_to_alias[name] != new_alias:
                    raise InvalidPropertyDictError(
                        f"{component.__class__.__name__} specifies multiple "
                        f"aliases for {name}: {name_to_alias[name]} in "
                        f"{other_properties_name} and "
                        f"{new_alias} in {cls.properties_name}."
                    )

                new_name = alias_to_name.setdefault(new_alias, name)
                if name != new_name:
                    raise InvalidPropertyDictError(
                        f"{component.__class__.__name__} maps two quantities "
                        f"to the same alias {new_alias}: {new_name} in "
                        f"{other_properties_name} and "
                        f"{name} in {cls.properties_name}."
                    )

    @classmethod
    def check(
        cls,
        component: "Component",
        other_properties_name: str = "input_properties",
    ) -> None:
        """Run all static checks on ``component``."""
        cls.check_component_is_initialized(component)
        cls.check_component_has_property(component)

        properties = StaticComponentOperator.factory(
            cls.properties_name
        ).get_properties(component)
        cls.check_property_type(component, properties)

        other_properties = StaticComponentOperator.factory(
            other_properties_name
        ).get_properties(component)
        cls.check_properties_have_units(
            component, properties, other_properties
        )
        cls.check_incompatible_units(
            component, properties, other_properties_name, other_properties
        )
        cls.check_properties_have_dims(component, properties, other_properties)
        cls.check_dims_like(component, properties, other_properties)
        cls.check_wildcard_dims(component, properties, other_properties)
        cls.check_match_dims_like(component, properties, other_properties)
        cls.check_aliases(
            component, properties, other_properties_name, other_properties
        )


class DiagnosticStaticComponentChecker(StaticComponentChecker):
    name = "diagnostic_properties"
    properties_name = "diagnostic_properties"


class InputStaticComponentChecker(StaticComponentChecker):
    name = "input_properties"
    properties_name = "input_properties"


class OutputStaticComponentChecker(StaticComponentChecker):
    name = "output_properties"
    properties_name = "output_properties"


class TendencyStaticComponentChecker(StaticComponentChecker):
    name = "tendency_properties"
    properties_name = "tendency_properties"

    @staticmethod
    def wrap_units(units: str) -> str:
        """
        Convert the units of a field into the units of the temporal variation
        of that field.
        """
        return units + " s^-1"
