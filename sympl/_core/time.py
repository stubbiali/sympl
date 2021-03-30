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
from datetime import datetime as real_datetime
from datetime import timedelta

from .exceptions import DependencyError

try:
    import cftime as ct

    if not all(
        hasattr(ct, attr)
        for attr in [
            "DatetimeNoLeap",
            "DatetimeProlepticGregorian",
            "DatetimeAllLeap",
            "Datetime360Day",
            "DatetimeJulian",
            "DatetimeGregorian",
        ]
    ):
        ct = None
except ImportError:
    ct = None


def datetime(
    year,
    month,
    day,
    hour=0,
    minute=0,
    second=0,
    microsecond=0,
    tzinfo=None,
    calendar="proleptic_gregorian",
):
    """
    Retrieves a datetime-like object with the requested calendar. Calendar types
    other than proleptic_gregorian require the netcdftime module to be
    installed.

    Parameters
    ----------
    year : int,
    month  : int,
    day  : int,
    hour  : int, optional
    minute  : int, optional
    second : int, optional
    microsecond : int, optional
    tzinfo  : datetime.tzinfo, optional
        A timezone informaton class, such as from pytz. Can only be used with
        'proleptic_gregorian' calendar, as netcdftime does not support
        timezones.
    calendar : string, optional
        Should be one of 'proleptic_gregorian', 'no_leap', '365_day',
        'all_leap', '366_day', '360_day', 'julian', or 'gregorian'. Default
        is 'proleptic_gregorian', which returns a normal Python datetime.
        Other options require the netcdftime module to be installed.

    Returns
    -------
    datetime : datetime-like
        The requested datetime. May be a Python datetime, or one of the
        datetime-like types in netcdftime.
    """
    kwargs = {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "microsecond": microsecond,
    }
    if calendar.lower() == "proleptic_gregorian":
        return real_datetime(tzinfo=tzinfo, **kwargs)
    elif tzinfo is not None:
        raise ValueError(
            "netcdftime does not support timezone-aware datetimes"
        )
    elif ct is None:
        raise DependencyError(
            "Calendars other than 'proleptic_gregorian' require the netcdftime "
            "package, which is not installed."
        )
    elif calendar.lower() in ("all_leap", "366_day"):
        return ct.DatetimeAllLeap(**kwargs)
    elif calendar.lower() in ("no_leap", "noleap", "365_day"):
        return ct.DatetimeNoLeap(**kwargs)
    elif calendar.lower() == "360_day":
        return ct.Datetime360Day(**kwargs)
    elif calendar.lower() == "julian":
        return ct.DatetimeJulian(**kwargs)
    elif calendar.lower() == "gregorian":
        return ct.DatetimeGregorian(**kwargs)


__all__ = (datetime, timedelta)
