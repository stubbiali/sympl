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
from dataclasses import dataclass, field
from datetime import datetime as real_datetime
import functools
import numpy as np
import time
import timeit
from typing import Dict, List, Optional

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

try:
    import cupy as cp

    # cp = None
except ImportError:
    cp = None


from sympl._core.data_array import DataArray
from sympl._core.exceptions import DependencyError


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


def datetime64_to_datetime(dt64):
    ts = (dt64 - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(
        1, "s"
    )
    return datetime.utcfromtimestamp(ts)


@dataclass
class Node:
    label: str
    parent: "Node" = None
    children: Dict[str, "Node"] = field(default_factory=dict)
    level: int = 0
    tic: float = 0
    total_calls: int = 0
    total_runtime: float = 0


class Timer:
    active: List[str] = []
    head: Optional[Node] = None
    tree: Dict[str, Node] = {}

    @classmethod
    def start(cls, label: str) -> None:
        # safe-guard
        if label in cls.active:
            return

        # mark node as active
        cls.active.append(label)

        # insert timer in the tree
        node_label = cls.active[0]
        node = cls.tree.setdefault(node_label, Node(node_label))
        for i, node_label in enumerate(cls.active[1:]):
            node = node.children.setdefault(
                node_label, Node(node_label, parent=node, level=i + 1)
            )
        cls.head = node

        # tic
        if cp is not None:
            try:
                cp.cuda.Device(0).synchronize()
            except RuntimeError:
                pass
        # cls.head.tic = timeit.default_timer()
        cls.head.tic = time.perf_counter()

    @classmethod
    def stop(cls, label: Optional[str] = None) -> None:
        # safe-guard
        if len(cls.active) == 0:
            return

        # only nested timers allowed!
        label = label or cls.active[-1]
        assert (
            label == cls.active[-1]
        ), f"Cannot stop {label} before stopping {cls.active[-1]}"

        # toc
        if cp is not None:
            try:
                cp.cuda.Device(0).synchronize()
            except RuntimeError:
                pass
        # toc = timeit.default_timer()
        toc = time.perf_counter()

        # update statistics
        cls.head.total_calls += 1
        cls.head.total_runtime += toc - cls.head.tic

        # mark node as not active
        cls.active = cls.active[:-1]

        # update head
        cls.head = cls.head.parent

    @classmethod
    def reset(cls) -> None:
        cls.active = []
        cls.head = None

        def cb(node):
            node.total_calls = 0
            node.total_runtime = 0

        for root in cls.tree.values():
            cls.traverse(cb, root)

    @classmethod
    def get_time(cls, label, units="ms") -> None:
        nodes = cls.get_nodes_from_label(label)
        assert len(nodes) > 0, f"{label} is not a valid timer identifier."

        raw_time = functools.reduce(
            lambda x, node: x + node.total_runtime, nodes, 0
        )
        time = (
            DataArray(raw_time, attrs={"units": "s"})
            .to_units(units)
            .data.item()
        )

        return time

    @classmethod
    def print(cls, label, units="ms") -> None:
        time = cls.get_time(label, units)
        print(f"{label}: {time:.3f} {units}")

    @classmethod
    def log(cls, logfile: str = "log.txt", units: str = "ms") -> None:
        # ensure all timers have been stopped
        assert len(cls.active) == 0, "Some timers are still running."

        # callback
        def cb(node, out, units, prefix="", has_peers=False):
            level = node.level
            prefix_now = prefix + "|- " if level > 0 else prefix
            time = (
                DataArray(node.total_runtime, attrs={"units": "s"})
                .to_units(units)
                .data.item()
            )
            out.write(f"{prefix_now}{node.label}: {time:.3f} {units}\n")

            prefix_new = (
                prefix
                if level == 0
                else prefix + "|  "
                if has_peers
                else prefix + "   "
            )
            peers_new = len(node.children)
            has_peers_new = peers_new > 0
            for i, label in enumerate(node.children.keys()):
                cb(
                    node.children[label],
                    out,
                    units,
                    prefix=prefix_new,
                    has_peers=has_peers_new and i < peers_new - 1,
                )

        # write to file
        with open(logfile, "w") as outfile:
            for root in cls.tree.values():
                cb(root, outfile, units)

    @staticmethod
    def traverse(cb, node, **kwargs) -> None:
        cb(node, **kwargs)
        for child in node.children.values():
            Timer.traverse(cb, child, **kwargs)

    @classmethod
    def get_nodes_from_label(cls, label) -> List[Node]:
        out = []

        def cb(node, out):
            if node.label == label:
                out.append(node)

        for root in cls.tree.values():
            Timer.traverse(cb, root, out=out)

        return out


class FakeTimer:
    @classmethod
    def start(cls, label: str) -> None:
        pass

    @classmethod
    def stop(cls, label: Optional[str] = None) -> None:
        pass
