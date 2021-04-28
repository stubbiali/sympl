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

try:
    from inspect import getfullargspec as getargspec
except ImportError:
    from inspect import getargspec

from sympl._core.utils import get_kwarg_defaults

if TYPE_CHECKING:
    pass


class MetaComponent(abc.ABCMeta):
    mcs_registry = []

    def __new__(mcs, cls_name, bases, cls_dict):
        cls = super().__new__(mcs, cls_name, bases, cls_dict)
        mcs.mcs_registry.append(cls)
        return cls

    def __instancecheck__(cls, instance):
        if issubclass(instance.__class__, tuple(cls.mcs_registry)):
            return True
        else:
            # checking if non-inheriting instance is a duck-type of a
            # component base class
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


class BaseComponent(abc.ABC, metaclass=MetaComponent):
    def __init__(self):
        self._initialized = True
