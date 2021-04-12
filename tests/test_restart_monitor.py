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
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from sympl import DataArray, InvalidDataArrayDictError, RestartMonitor

random = np.random.RandomState(0)

nx = 5
ny = 5
nz = 3
state = {
    "time": datetime(2013, 7, 20),
    "air_temperature": DataArray(
        random.randn(nx, ny, nz),
        dims=["lon", "lat", "mid_levels"],
        attrs={"units": "degK"},
    ),
    "air_pressure": DataArray(
        random.randn(nx, ny, nz),
        dims=["lon", "lat", "mid_levels"],
        attrs={"units": "Pa"},
    ),
}


def test_restart_monitor_initializes():
    restart_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "restart.nc"
    )
    if os.path.isfile(restart_filename):
        os.remove(restart_filename)
    assert not os.path.isfile(restart_filename)
    RestartMonitor(restart_filename)
    assert not os.path.isfile(
        restart_filename
    )  # should not create file on init


def test_restart_monitor_stores_state():
    restart_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "restart.nc"
    )
    if os.path.isfile(restart_filename):
        os.remove(restart_filename)
    assert not os.path.isfile(restart_filename)
    monitor = RestartMonitor(restart_filename)
    assert not os.path.isfile(
        restart_filename
    )  # should not create file on init
    monitor.store(state)
    assert os.path.isfile(restart_filename)
    new_monitor = RestartMonitor(restart_filename)
    loaded_state = new_monitor.load()
    for name in state.keys():
        if name is "time":
            assert state["time"] == loaded_state["time"]
        else:
            assert np.all(state[name].values == loaded_state[name].values)
            assert state[name].dims == loaded_state[name].dims
            assert state[name].attrs == loaded_state[name].attrs


if __name__ == "__main__":
    pytest.main([__file__])
