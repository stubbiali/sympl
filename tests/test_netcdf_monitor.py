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
import unittest
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from sympl import DataArray, InvalidDataArrayDictError, NetCDFMonitor

random = np.random.RandomState(0)

nx = 5
ny = 5
nz = 3
state = {
    "time": datetime(2013, 7, 20),
    "air_temperature": DataArray(
        random.randn(nx, ny, nz),
        dims=["lon", "lat", "mid_levels"],
        attrs={"units": "degK", "long_name": "air_temperature"},
    ),
    "air_pressure": DataArray(
        random.randn(nx, ny, nz),
        dims=["lon", "lat", "mid_levels"],
        attrs={"units": "Pa", "long_name": "air_pressure"},
    ),
}


class NetCDFMonitorAliasTests(unittest.TestCase):
    def setUp(self):
        self.ncfile = "out.nc"

    def tearDown(self):
        if os.path.isfile(self.ncfile):
            os.remove(self.ncfile)

    def store_state_and_check_file(self, aliases):
        assert not os.path.isfile(self.ncfile)
        monitor = NetCDFMonitor(
            self.ncfile, aliases=aliases, write_on_store=True
        )
        monitor.store(state)
        assert os.path.isfile(self.ncfile)

    def check_nc_var(self, varname, varunit, longname):
        with xr.open_dataset(self.ncfile) as ds:
            assert len(ds.data_vars.keys()) == 2
            assert varname in ds.data_vars.keys()
            assert ds.data_vars[varname].attrs["units"] == varunit
            assert ds.data_vars[varname].attrs["long_name"] == longname
            assert tuple(ds.data_vars[varname].shape) == (1, nx, ny, nz)

    def test_state_key_emptystring(self):
        aliases = {"air_temperature": "T"}
        bad_state = {
            "time": datetime(2013, 7, 20),
            "": DataArray(
                random.randn(nx, ny, nz),
                dims=["lon", "lat", "mid_levels"],
                attrs={"units": "degK", "long_name": "air_temperature"},
            ),
        }
        assert not os.path.isfile(self.ncfile)
        monitor = NetCDFMonitor(
            self.ncfile, aliases=aliases, write_on_store=True
        )
        self.assertRaises(ValueError, monitor.store, bad_state)
        assert not os.path.isfile(self.ncfile)

    def test_keys_string_values_string(self):
        aliases = {"air_temperature": "T"}
        self.store_state_and_check_file(aliases)
        self.check_nc_var("T", "degK", "air_temperature")
        self.check_nc_var("air_pressure", "Pa", "air_pressure")

    def test_keys_nonstring_values_string(self):
        aliases = {1.0: "T"}
        assert not os.path.isfile(self.ncfile)
        self.assertRaises(
            TypeError,
            NetCDFMonitor,
            self.ncfile,
            aliases=aliases,
            write_on_store=True,
        )
        assert not os.path.isfile(self.ncfile)

    def test_keys_string_values_nonstring(self):
        aliases = {"air_temperature": 1.0}
        assert not os.path.isfile(self.ncfile)
        self.assertRaises(
            TypeError,
            NetCDFMonitor,
            self.ncfile,
            aliases=aliases,
            write_on_store=True,
        )
        assert not os.path.isfile(self.ncfile)

    def test_keys_string_values_emptystring(self):
        # this SHOULD raise a ValueError
        aliases = {"air_temperature": ""}
        assert not os.path.isfile(self.ncfile)
        monitor = NetCDFMonitor(
            self.ncfile, aliases=aliases, write_on_store=True
        )
        self.assertRaises(ValueError, monitor.store, state)
        assert not os.path.isfile(self.ncfile)

    def test_keys_partialstring_values_emptystring(self):
        # this SHOULD NOT raise a ValueError
        aliases = {"air_": ""}
        self.store_state_and_check_file(aliases)
        self.check_nc_var("temperature", "degK", "air_temperature")
        self.check_nc_var("pressure", "Pa", "air_pressure")

    def test_empty_aliases(self):
        aliases = {}
        self.store_state_and_check_file(aliases)
        self.check_nc_var("air_temperature", "degK", "air_temperature")
        self.check_nc_var("air_pressure", "Pa", "air_pressure")

    def test_two_aliases(self):
        aliases = {"air_temperature": "T", "air_pressure": "P"}
        self.store_state_and_check_file(aliases)
        self.check_nc_var("T", "degK", "air_temperature")
        self.check_nc_var("P", "Pa", "air_pressure")


def test_netcdf_monitor_initializes():
    assert not os.path.isfile("out.nc")
    NetCDFMonitor("out.nc")
    assert not os.path.isfile(
        "out.nc"
    )  # should not create output file on init


def test_netcdf_monitor_initializes_with_kwargs():
    assert not os.path.isfile("out.nc")
    NetCDFMonitor(
        "out.nc",
        time_units="hours",
        store_names=("air_temperature", "air_pressure"),
        write_on_store=True,
    )
    assert not os.path.isfile(
        "out.nc"
    )  # should not create output file on init


def test_netcdf_monitor_single_time_all_vars():
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc")
        monitor.store(state)
        assert not os.path.isfile("out.nc")  # not set to write on store
        monitor.write()
        assert os.path.isfile("out.nc")
        with xr.open_dataset("out.nc") as ds:
            assert len(ds.data_vars.keys()) == 2
            assert "air_temperature" in ds.data_vars.keys()
            assert ds.data_vars["air_temperature"].attrs["units"] == "degK"
            assert tuple(ds.data_vars["air_temperature"].shape) == (
                1,
                nx,
                ny,
                nz,
            )
            assert "air_pressure" in ds.data_vars.keys()
            assert ds.data_vars["air_pressure"].attrs["units"] == "Pa"
            assert tuple(ds.data_vars["air_pressure"].shape) == (1, nx, ny, nz)
            assert len(ds["time"]) == 1
            assert ds["time"][0] == np.datetime64(state["time"])
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


def test_netcdf_monitor_multiple_times_batched_all_vars():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc")
        for time in time_list:
            current_state["time"] = time
            monitor.store(current_state)
            assert not os.path.isfile("out.nc")  # not set to write on store
        monitor.write()
        assert os.path.isfile("out.nc")
        with xr.open_dataset("out.nc") as ds:
            assert len(ds.data_vars.keys()) == 2
            assert "air_temperature" in ds.data_vars.keys()
            assert ds.data_vars["air_temperature"].attrs["units"] == "degK"
            assert tuple(ds.data_vars["air_temperature"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert "air_pressure" in ds.data_vars.keys()
            assert ds.data_vars["air_pressure"].attrs["units"] == "Pa"
            assert tuple(ds.data_vars["air_pressure"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert len(ds["time"]) == len(time_list)
            assert np.all(
                ds["time"].values
                == [np.datetime64(time) for time in time_list]
            )
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


def test_netcdf_monitor_multiple_times_sequential_all_vars():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc")
        for time in time_list:
            current_state["time"] = time
            monitor.store(current_state)
            monitor.write()
        assert os.path.isfile("out.nc")
        with xr.open_dataset("out.nc") as ds:
            assert len(ds.data_vars.keys()) == 2
            assert "air_temperature" in ds.data_vars.keys()
            assert ds.data_vars["air_temperature"].attrs["units"] == "degK"
            assert tuple(ds.data_vars["air_temperature"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert "air_pressure" in ds.data_vars.keys()
            assert ds.data_vars["air_pressure"].attrs["units"] == "Pa"
            assert tuple(ds.data_vars["air_pressure"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert len(ds["time"]) == len(time_list)
            assert np.all(
                ds["time"].values
                == [np.datetime64(time) for time in time_list]
            )
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


def test_netcdf_monitor_multiple_times_sequential_all_vars_timedelta():
    time_list = [
        timedelta(hours=0),
        timedelta(hours=6),
        timedelta(hours=12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc")
        for time in time_list:
            current_state["time"] = time
            monitor.store(current_state)
            monitor.write()
        assert os.path.isfile("out.nc")
        with xr.open_dataset("out.nc") as ds:
            assert len(ds.data_vars.keys()) == 2
            assert "air_temperature" in ds.data_vars.keys()
            assert ds.data_vars["air_temperature"].attrs["units"] == "degK"
            assert tuple(ds.data_vars["air_temperature"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert "air_pressure" in ds.data_vars.keys()
            assert ds.data_vars["air_pressure"].attrs["units"] == "Pa"
            assert tuple(ds.data_vars["air_pressure"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert len(ds["time"]) == len(time_list)
            assert np.all(
                ds["time"].values
                == [np.timedelta64(time) for time in time_list]
            )
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


def test_netcdf_monitor_multiple_times_batched_single_var():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc", store_names=["air_temperature"])
        for time in time_list:
            current_state["time"] = time
            monitor.store(current_state)
            assert not os.path.isfile("out.nc")  # not set to write on store
        monitor.write()
        assert os.path.isfile("out.nc")
        with xr.open_dataset("out.nc") as ds:
            assert len(ds.data_vars.keys()) == 1
            assert "air_temperature" in ds.data_vars.keys()
            assert ds.data_vars["air_temperature"].attrs["units"] == "degK"
            assert tuple(ds.data_vars["air_temperature"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert len(ds["time"]) == len(time_list)
            assert np.all(
                ds["time"].values
                == [np.datetime64(time) for time in time_list]
            )
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


def test_netcdf_monitor_multiple_times_sequential_single_var():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc", store_names=["air_temperature"])
        for time in time_list:
            current_state["time"] = time
            monitor.store(current_state)
            monitor.write()
        assert os.path.isfile("out.nc")
        with xr.open_dataset("out.nc") as ds:
            assert len(ds.data_vars.keys()) == 1
            assert "air_temperature" in ds.data_vars.keys()
            assert ds.data_vars["air_temperature"].attrs["units"] == "degK"
            assert tuple(ds.data_vars["air_temperature"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert len(ds["time"]) == len(time_list)
            assert np.all(
                ds["time"].values
                == [np.datetime64(time) for time in time_list]
            )
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


def test_netcdf_monitor_single_write_on_store():
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc", write_on_store=True)
        monitor.store(state)
        assert os.path.isfile("out.nc")
        with xr.open_dataset("out.nc") as ds:
            assert len(ds.data_vars.keys()) == 2
            assert "air_temperature" in ds.data_vars.keys()
            assert ds.data_vars["air_temperature"].attrs["units"] == "degK"
            assert tuple(ds.data_vars["air_temperature"].shape) == (
                1,
                nx,
                ny,
                nz,
            )
            assert "air_pressure" in ds.data_vars.keys()
            assert ds.data_vars["air_pressure"].attrs["units"] == "Pa"
            assert tuple(ds.data_vars["air_pressure"].shape) == (1, nx, ny, nz)
            assert len(ds["time"]) == 1
            assert ds["time"][0] == np.datetime64(state["time"])
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


def test_netcdf_monitor_multiple_write_on_store():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc", write_on_store=True)
        for time in time_list:
            current_state["time"] = time
            monitor.store(current_state)
        assert os.path.isfile("out.nc")
        with xr.open_dataset("out.nc") as ds:
            assert len(ds.data_vars.keys()) == 2
            assert "air_temperature" in ds.data_vars.keys()
            assert ds.data_vars["air_temperature"].attrs["units"] == "degK"
            assert tuple(ds.data_vars["air_temperature"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert "air_pressure" in ds.data_vars.keys()
            assert ds.data_vars["air_pressure"].attrs["units"] == "Pa"
            assert tuple(ds.data_vars["air_pressure"].shape) == (
                len(time_list),
                nx,
                ny,
                nz,
            )
            assert len(ds["time"]) == len(time_list)
            assert np.all(
                ds["time"].values
                == [np.datetime64(time) for time in time_list]
            )
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


def test_netcdf_monitor_raises_when_names_change_on_sequential_write():
    current_state = state.copy()
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc")
        current_state["time"] = datetime(2013, 7, 20, 0)
        monitor.store(current_state)
        monitor.write()
        assert os.path.isfile("out.nc")
        current_state["time"] = datetime(2013, 7, 20, 6)
        current_state["air_density"] = current_state["air_pressure"]
        monitor.store(current_state)
        try:
            monitor.write()
        except InvalidDataArrayDictError:
            pass
        except Exception as err:
            raise err
        else:
            raise AssertionError(
                "Expected InvalidStateError but was not raised."
            )
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


def test_netcdf_monitor_raises_when_names_change_on_batch_write():
    current_state = state.copy()
    try:
        assert not os.path.isfile("out.nc")
        monitor = NetCDFMonitor("out.nc")
        current_state["time"] = datetime(2013, 7, 20, 0)
        monitor.store(current_state)
        assert not os.path.isfile("out.nc")
        current_state["time"] = datetime(2013, 7, 20, 6)
        current_state["air_density"] = current_state["air_pressure"]
        monitor.store(current_state)
        try:
            monitor.write()
        except InvalidDataArrayDictError:
            pass
        except Exception as err:
            raise err
        else:
            raise AssertionError(
                "Expected InvalidStateError but was not raised."
            )
    finally:  # make sure we remove the output file
        if os.path.isfile("out.nc"):
            os.remove("out.nc")


if __name__ == "__main__":
    pytest.main([__file__])
