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
import xarray as xr
from pint.errors import DimensionalityError

from .units import data_array_to_units as to_units_function


class DataArray(xr.DataArray):
    __slots__ = []

    def __add__(self, other):
        """If this DataArray is on the left side of the addition, keep its
        attributes when adding to the other object."""
        result = super(DataArray, self).__add__(other)
        result.attrs = self.attrs
        return result

    def __sub__(self, other):
        """If this DataArray is on the left side of the subtraction, keep its
        attributes when subtracting the other object."""
        result = super(DataArray, self).__sub__(other)
        result.attrs = self.attrs
        return result

    def to_units(self, units):
        """
        Convert the units of this DataArray, if necessary. No conversion is
        performed if the units are the same as the units of this DataArray.
        The units of this DataArray are determined from the "units" attribute in
        attrs.

        Args
        ----
        units : str
            The desired units.

        Raises
        ------
        ValueError
            If the units are invalid for this object.
        KeyError
            If this object does not have units information in its attrs.

        Returns
        -------
        converted_data : DataArray
            A DataArray containing the data from this object in the
            desired units, if possible.
        """
        if "units" not in self.attrs:
            raise KeyError('"units" not present in attrs')
        try:
            return to_units_function(self, units)
        except DimensionalityError as err:
            raise ValueError(str(err))
