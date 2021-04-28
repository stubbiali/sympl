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
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING

from sympl._core.data_array import DataArray
from sympl._core.factory import AbstractFactory
from sympl._core.static_operators import StaticComponentOperator

if TYPE_CHECKING:
    from sympl._core.typingx import (
        Array,
        ArrayDict,
        Component,
        DataArrayDict,
        NDArrayLike,
        NDArrayLikeDict,
    )


class DynamicComponentOperator(abc.ABC):
    properties_name: str = None

    def __init__(self, component: "Component") -> None:
        self.properties = getattr(component, self.properties_name, {})
        self.input_properties = getattr(component, "input_properties", {})
        self.static_operator = StaticComponentOperator.factory(
            self.properties_name
        )
        self.aliases = self.static_operator.get_aliases(component)
        self.dims = self.static_operator.get_dims(component)

    def get_dim_lengths(
        self, dataarray_dict: "DataArrayDict"
    ) -> Dict[str, Set[int]]:
        """Get the shape of the fields along each dimension."""
        out = {}
        for name in self.properties:
            field = self.get_field(name, dataarray_dict)
            if field is not None:
                for idx, dim in enumerate(field.dims):
                    s = out.setdefault(dim, set())
                    s.add(field.shape[idx])
        return out

    def get_field(
        self,
        name: str,
        array_dict: "ArrayDict",
        input_array_dict: Optional["ArrayDict"] = None,
    ) -> Optional["Array"]:
        """Get the array for the field ``name``."""
        if name in array_dict:
            return array_dict[name]
        if name in self.aliases and self.aliases[name] in array_dict:
            return array_dict[self.aliases[name]]
        if (
            self.properties_name != "input_properties"
            and input_array_dict is not None
        ):
            return InflowComponentOperator.factory(
                "input_properties"
            ).get_field(name, input_array_dict)
        return None

    def get_wildcard_dims(
        self,
        name: str,
        dataarray_dict: "DataArrayDict",
        input_dataarray_dict: Optional["DataArrayDict"] = None,
    ) -> List[str]:
        """
        Get the actual dimensions of ``name`` matching the wildcard character.
        """
        if "dims_like" in self.properties[name] or (
            "dims" in self.properties[name]
            and "*" not in self.properties[name]["dims"]
        ):
            return []
        elif "match_dims_like" in self.properties[name]:
            return self.get_wildcard_dims(
                self.properties[name]["match_dims_like"],
                dataarray_dict,
                input_dataarray_dict,
            )
        else:
            field = self.get_field(name, dataarray_dict, input_dataarray_dict)
            return [
                str(dim)
                for dim in field.dims
                if field is not None
                and dim not in self.properties[name]["dims"]
            ]

    def get_target_field_dims(
        self,
        name,
        dataarray_dict: "DataArrayDict",
        input_dataarray_dict: Optional["DataArrayDict"] = None,
    ) -> Sequence[str]:
        """Get the target dimensions for a field."""
        field_properties = self.properties.get(name, {})
        if "dims_like" in field_properties:
            name_like = field_properties["dims_like"]
            return (
                field_properties["dims"]
                if name_like in self.properties
                else self.input_properties[name_like]["dims"]
            )
        elif "dims" in field_properties:
            if "*" in field_properties["dims"]:
                wildcard_dims = self.get_wildcard_dims(
                    name, dataarray_dict, input_dataarray_dict
                )
                out = []
                for dim in field_properties["dims"]:
                    if dim == "*":
                        out += wildcard_dims
                    else:
                        out.append(dim)
            else:
                out = field_properties["dims"]
            return out
        else:
            # virtually, we should never enter this branch...
            field_like = self.get_field(name, input_dataarray_dict or {})
            return field_like.dims if field_like is not None else []

    def get_target_dims(
        self,
        dataarray_dict: "DataArrayDict",
        input_dataarray_dict: Optional["DataArrayDict"] = None,
    ) -> Dict[str, Sequence[str]]:
        """
        Get the target dimensions for each sensible field in ``dataarray_dict``.
        """
        out = {}

        for name in self.properties:
            if "dims_like" in self.properties[name]:
                name_like = self.properties[name]["dims_like"]
                out[name] = (
                    self.properties[name_like]["dims"]
                    if name_like in self.properties
                    else self.input_properties[name_like]["dims"]
                )
            elif "dims" in self.properties[name]:
                if "*" in self.properties[name]["dims"]:
                    wildcard_dims = self.get_wildcard_dims(
                        name, dataarray_dict, input_dataarray_dict
                    )
                    out[name] = []
                    for dim in self.properties[name]["dims"]:
                        if dim == "*":
                            out[name] += wildcard_dims
                        else:
                            out[name].append(dim)
                else:
                    out[name] = self.properties[name]["dims"]
            else:
                # virtually, we should never enter this branch...
                field_like = self.get_field(name, input_dataarray_dict or {})
                out[name] = field_like.dims if field_like is not None else []

        return out


class InflowComponentOperator(DynamicComponentOperator, AbstractFactory):
    def get_actual_dims(
        self, dataarray_dict: "DataArrayDict"
    ) -> Dict[str, Sequence[str]]:
        """
        Get the actual dimensions for each sensible field in ``dataarray_dict``.
        """
        out = {}
        for name in self.properties:
            field = self.get_field(name, dataarray_dict)
            if field is not None:
                out[name] = field.dims
        return out

    def get_ndarray_dict(
        self, dataarray_dict: "DataArrayDict"
    ) -> "NDArrayLikeDict":
        """Extract the raw data from the DataArrays."""
        target_dims = self.get_target_dims(dataarray_dict)
        out = {}
        if "time" in dataarray_dict:
            out["time"] = dataarray_dict["time"]
        for name in self.properties:
            raw_name = (
                self.properties[name]["alias"]
                if "alias" in self.properties[name]
                else name
            )
            da = self.get_field(name, dataarray_dict)
            if da is not None:
                if not all(
                    d1 == d2 for d1, d2 in zip(da.dims, target_dims[name])
                ):
                    da = da.transpose(*target_dims[name])
                out[raw_name] = da.data
        return out


class InputComponentOperator(InflowComponentOperator):
    name = "input_properties"
    properties_name = "input_properties"


class DiagnosticInflowComponentOperator(InflowComponentOperator):
    name = "diagnostic_properties"
    properties_name = "diagnostic_properties"


class OutputInflowComponentOperator(InflowComponentOperator):
    name = "output_properties"
    properties_name = "output_properties"


class TendencyInflowComponentOperator(InflowComponentOperator):
    name = "tendency_properties"
    properties_name = "tendency_properties"


class OutflowComponentOperator(DynamicComponentOperator, AbstractFactory):
    def get_dataarray(
        self,
        name: str,
        ndarray: "NDArrayLike",
        input_dataarray_dict: "DataArrayDict",
    ) -> "DataArray":
        """Wrap a raw array into a DataArray."""
        target_dims = self.get_target_field_dims(name, input_dataarray_dict)
        return DataArrayDict(
            ndarray,
            dims=target_dims,
            attrs={"units": self.properties[name]["units"]},
        )

    def get_dataarray_dict(
        self,
        ndarray_dict: "NDArrayLikeDict",
        input_dataarray_dict: "DataArrayDict",
        out: Optional["DataArrayDict"] = None,
    ) -> "DataArrayDict":
        """Wrap raw arrays into DataArrays."""
        target_dims = self.get_target_dims(input_dataarray_dict)
        out = out if out is not None else {}
        if "time" in ndarray_dict:
            out["time"] = ndarray_dict["time"]
        for name in self.properties:
            raw_name = (
                self.properties[name]["alias"]
                if "alias" in self.properties[name]
                else name
            )
            data = self.get_field(name, ndarray_dict)
            if data is not None:
                out[raw_name] = DataArray(
                    data,
                    dims=target_dims[name],
                    attrs={"units": self.properties[name]["units"]},
                )
        return out


class DiagnosticOutflowComponentOperator(OutflowComponentOperator):
    name = "diagnostic_properties"
    properties_name = "diagnostic_properties"


class OutputOutflowComponentOperator(OutflowComponentOperator):
    name = "output_properties"
    properties_name = "output_properties"


class TendencyOutflowComponentOperator(OutflowComponentOperator):
    name = "tendency_properties"
    properties_name = "tendency_properties"
