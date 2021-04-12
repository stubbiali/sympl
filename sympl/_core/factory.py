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
from typing import Any, Dict, Tuple, Type

from sympl._core.exceptions import FactoryError


class MetaFactory(type):
    def __new__(
        mcs, cls_name: str, bases: Tuple[Type], cls_dict: Dict[str, Any]
    ):
        if cls_name == "Factory" or bases[-1].__name__ == "Factory":
            return super().__new__(mcs, cls_name, bases, cls_dict)
        else:
            parent_cls = bases[-1]
            registry = getattr(parent_cls, "registry", {})
            subregistry = registry.setdefault(
                parent_cls.__module__ + "." + parent_cls.__name__, {}
            )
            name = cls_dict.setdefault("name", "default")
            if name in subregistry:
                raise FactoryError(
                    f"Cannot register {cls_name} as {name} since another "
                    f"class ({subregistry[name].__name__}) has already been "
                    f"registered with that name."
                )
            cls = super().__new__(mcs, cls_name, bases, cls_dict)
            subregistry[name] = cls
            return cls


class Factory(metaclass=MetaFactory):
    name = "default"
    registry: Dict[str, "Factory"] = {}

    @classmethod
    def factory(cls, name: str, *args, **kwargs):
        subregistry = cls.registry.get(cls.__module__ + "." + cls.__name__, {})
        child_cls = subregistry.get(name, None)
        if child_cls is None:
            raise FactoryError(
                f"No class inheriting {cls.__name__} registered under {name}."
            )
        return child_cls(*args, **kwargs)


class MetaAbstractFactory(abc.ABCMeta):
    def __new__(
        mcs, cls_name: str, bases: Tuple[Type], cls_dict: Dict[str, Any]
    ):
        if (
            cls_name == "AbstractFactory"
            or bases[-1].__name__ == "AbstractFactory"
        ):
            return super().__new__(mcs, cls_name, bases, cls_dict)
        else:
            parent_cls = bases[-1]
            registry = getattr(parent_cls, "registry", {})
            subregistry = registry.setdefault(
                parent_cls.__module__ + "." + parent_cls.__name__, {}
            )
            name = cls_dict.setdefault("name", "default")
            if name in subregistry:
                raise FactoryError(
                    f"Cannot register {cls_name} as {name} since another "
                    f"class ({subregistry[name].__name__}) has already been "
                    f"registered with that name."
                )
            cls = super().__new__(mcs, cls_name, bases, cls_dict)
            subregistry[name] = cls
            return cls


class AbstractFactory(metaclass=MetaAbstractFactory):
    name = "default"
    registry: Dict[str, "AbstractFactory"] = {}

    @classmethod
    def factory(cls, name: str, *args, **kwargs):
        subregistry = cls.registry.get(cls.__module__ + "." + cls.__name__, {})
        child_cls = subregistry.get(name, None)
        if child_cls is None:
            raise FactoryError(
                f"No class inheriting {cls.__name__} registered under {name}."
            )
        return child_cls(*args, **kwargs)
