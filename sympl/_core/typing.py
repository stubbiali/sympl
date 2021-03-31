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
from datetime import datetime
from typing import Any, Dict, Sequence, TypeVar, Union

from sympl._core.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    Monitor,
    Stepper,
    TendencyComponent,
)
from sympl._core.data_array import DataArray
from sympl._core.time import datetime as sympl_datetime


Component = Union[
    DiagnosticComponent,
    ImplicitTendencyComponent,
    Monitor,
    Stepper,
    TendencyComponent,
]
DateTime = Union[datetime, sympl_datetime]
DataArrayDict = Dict[str, Union[DateTime, DataArray]]
NDArrayLike = TypeVar("NDArrayLike")
NDArrayLikeDict = Dict[str, Union[DateTime, NDArrayLike]]
Property = Dict[str, Union[str, Sequence[str]]]
PropertyDict = Dict[str, Property]
