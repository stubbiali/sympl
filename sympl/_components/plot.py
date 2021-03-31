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
from sympl._core.base_components import Monitor
from sympl._core.data_array import DataArray
from sympl._core.exceptions import DependencyError


def copy_state(state):
    return_state = {}
    for name, quantity in state.items():
        if isinstance(quantity, DataArray):
            return_state[name] = DataArray(
                quantity.values.copy(),
                quantity.coords,
                quantity.dims,
                quantity.name,
                quantity.attrs,
            )
        else:
            return_state[name] = quantity
    return return_state


class PlotFunctionMonitor(Monitor):
    """
    A Monitor which uses a user-defined function to draw figures using model
    state.
    """

    def __init__(self, plot_function, interactive=True):
        """
        Initialize a PlotFunctionMonitor.

        Args
        ----
        plot_function : func
            A function plot_function(fig, state) that
            draws the given state onto the given (initially clear) figure.
        interactive: bool, optional
            If true, matplotlib's interactive mode will be enabled,
            allowing plot animation while other computation is running.
        """
        global plt
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise DependencyError(
                "matplotlib must be installed to use PlotFunctionMonitor"
            )
        if interactive:
            plt.ion()
            self._fig = plt.figure()
        else:
            plt.ioff()
            self._fig = None
        self._plot_function = plot_function

    @property
    def interactive(self):
        return self._fig is not None

    def store(self, state):
        """
        Updates the plot using the given state.

        Args
        ----
        state : dict
            A model state dictionary.
        """
        if self.interactive:
            self._fig.clear()
            fig = self._fig
        else:
            fig = plt.figure()

        self._plot_function(fig, copy_state(state))

        fig.canvas.draw()
        if not self.interactive:
            plt.show()
