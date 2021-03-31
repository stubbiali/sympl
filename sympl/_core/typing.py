# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Any, Dict, Sequence, TypeVar, Union

from sympl._core.data_array import DataArray
from sympl._core.time import datetime as sympl_datetime


DateTime = Union[datetime, sympl_datetime]
DataArrayDict = Dict[str, Union[DateTime, DataArray]]
NDArrayLike = TypeVar("NDArrayLike")
NDArrayLikeDict = Dict[str, Union[DateTime, NDArrayLike]]
Property = Dict[str, Union[str, Sequence[str]]]
PropertyDict = Dict[str, Property]
