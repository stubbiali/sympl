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
from sympl._core.data_array import DataArray
from sympl._core.state import add, copy_untouched_quantities, multiply
from sympl._core.tendency_stepper import TendencyStepper


class SSPRungeKutta(TendencyStepper):
    """
    A TendencyStepper using the Strong Stability Preserving Runge-Kutta scheme,
    as in Numerical Methods for Fluid Dynamics by Dale Durran (2nd ed) and
    as proposed by Shu and Osher (1988).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a strong stability preserving Runge-Kutta time stepper.

        Args
        ----
        *args : TendencyComponent or ImplicitTendencyComponent
            Objects to call for tendencies when doing time stepping.
        stages: int, optional
            Number of stages to use. Should be 2 or 3. Default is 3.
        """
        stages = kwargs.pop("stages", 3)
        if stages not in (2, 3):
            raise ValueError(
                "stages must be one of 2 or 3, received {}".format(stages)
            )
        self._stages = stages
        self._euler_stepper = AdamsBashforth(*args, order=1)
        super(SSPRungeKutta, self).__init__(*args, **kwargs)

    def _call(self, state, timestep):
        """
        Updates the input state dictionary and returns a new state corresponding
        to the next timestep.

        Args
        ----
        state : dict
            The current model state. Will be updated in-place by
            the call with any diagnostics from the current timestep.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state.
        new_state : dict
            The model state at the next timestep.
        """
        if self._stages == 2:
            return self._step_2_stages(state, timestep)
        elif self._stages == 3:
            return self._step_3_stages(state, timestep)

    def _step_3_stages(self, state, timestep):
        diagnostics, state_1 = self._euler_stepper(state, timestep)
        _, state_1_5 = self._euler_stepper(state_1, timestep)
        state_2 = add(multiply(0.75, state), multiply(0.25, state_1_5))
        _, state_2_5 = self._euler_stepper(state_2, timestep)
        out_state = add(multiply(1.0 / 3, state), multiply(2.0 / 3, state_2_5))
        return diagnostics, out_state

    def _step_2_stages(self, state, timestep):
        assert state is not None
        diagnostics, state_1 = self._euler_stepper(state, timestep)
        assert state_1 is not None
        _, state_2 = self._euler_stepper(state_1, timestep)
        out_state = multiply(0.5, add(state, state_2))
        return diagnostics, out_state


class AdamsBashforth(TendencyStepper):
    """A TendencyStepper using the Adams-Bashforth scheme."""

    def __init__(self, *args, **kwargs):
        """
        Initialize an Adams-Bashforth time stepper.

        Args
        ----
        *args : TendencyComponent or ImplicitTendencyComponent
            Objects to call for tendencies when doing time stepping.
        order : int, optional
            The order of accuracy to use. Must be between
            1 and 4. 1 is the same as the Euler method. Default is 3.
        """
        order = kwargs.pop("order", 3)
        if isinstance(order, float) and order.is_integer():
            order = int(order)
        if not isinstance(order, int):
            raise TypeError("order must be an integer")
        if not 1 <= order <= 4:
            raise ValueError("order must be between 1 and 4")
        self._order = order
        self._timestep = None
        self._tendencies_list = []
        super(AdamsBashforth, self).__init__(*args, **kwargs)

    def _call(self, state, timestep):
        """
        Updates the input state dictionary and returns a new state corresponding
        to the next timestep.

        Args
        ----
        state : dict
            The current model state. Will be updated in-place by
            the call with any diagnostics from the current timestep.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state.
        new_state : dict
            The model state at the next timestep.

        Raises
        ------
        ValueError
            If the timestep is not the same as the last time
            step() was called on this instance of this object.
        """
        self._ensure_constant_timestep(timestep)
        state = state.copy()
        tendencies, diagnostics = self.prognostic(state, timestep)
        convert_tendencies_units_for_state(tendencies, state)
        self._tendencies_list.append(tendencies)
        new_state = self._perform_step(state, timestep)
        copy_untouched_quantities(state, new_state)
        if len(self._tendencies_list) == self._order:
            self._tendencies_list.pop(0)  # remove the oldest entry
        return diagnostics, new_state

    def _perform_step(self, state, timestep):
        # if we don't have enough previous tendencies built up, use lower order
        order = min(self._order, len(self._tendencies_list))
        if order == 1:
            new_state = step_forward_euler(
                state, self._tendencies_list[-1], timestep
            )
        elif order == 2:
            new_state = second_bashforth(
                state, self._tendencies_list, timestep
            )
        elif order == 3:
            new_state = third_bashforth(state, self._tendencies_list, timestep)
        elif order == 4:
            new_state = fourth_bashforth(
                state, self._tendencies_list, timestep
            )
        else:
            # the following should never happen, if it is there's a bug
            raise RuntimeError("order should be integer between 1 and 4")
        return new_state

    def _ensure_constant_timestep(self, timestep):
        if self._timestep is None:
            self._timestep = timestep
        elif self._timestep != timestep:
            raise ValueError(
                "timestep must be constant for Adams-Bashforth time stepping"
            )


def convert_tendencies_units_for_state(tendencies, state):
    """
    Converts the units of any DataArrays with unit informaton in the
    tendencies dictionary to have units of {value_units}/second where
    {value_units} is the units of the value in the state dictionary.
    This is done in-place.
    """
    for quantity_name in tendencies.keys():
        if isinstance(tendencies[quantity_name], DataArray) and (
            "units" in tendencies[quantity_name].attrs
        ):
            desired_units = "{} s^-1".format(
                state[quantity_name].attrs["units"]
            )
            tendencies[quantity_name] = tendencies[quantity_name].to_units(
                desired_units
            )


class Leapfrog(TendencyStepper):
    """A TendencyStepper using the Leapfrog scheme.

    This scheme calculates the
    values at time $t_{n+1}$ using the derivatives at $t_{n}$ and values at
    $t_{n-1}$. Following the step, an Asselin filter is applied to damp the
    computational mode that results from the scheme and maintain stability. The
    Asselin filter brings the values at $t_{n}$ (and optionally the values at
    $t_{n+1}$, according to Williams (2009)) closer to the mean of the values
    at $t_{n-1}$ and $t_{n+1}$."""

    def __init__(self, *args, **kwargs):
        """
        Initialize a Leapfrog time stepper.

        Args
        ----
        *args : TendencyComponent or ImplicitTendencyComponent
            Objects to call for tendencies when doing time stepping.
        asselin_strength : float, optional
            The filter parameter used to determine the strength
            of the Asselin filter. Default is 0.05.
        alpha : float, optional
            Constant from Williams (2009), where the midpoint is shifted
            by alpha*influence, and the right point is shifted
            by (1-alpha)*influence. If alpha is 1 then the behavior
            is that of the classic Robert-Asselin time filter, while if it
            is 0.5 the filter will conserve the three-point mean.
            Default is 0.5.

        References
        ----------
        Williams, P., 2009: A Proposed Modification to the Robert-Asselin
        Time Filter. Mon. Wea. Rev., 137, 2538--2546,
        doi: 10.1175/2009MWR2724.1.
        """
        self._old_state = None
        self._asselin_strength = kwargs.pop("asselin_strength", 0.05)
        self._timestep = None
        self._alpha = kwargs.pop("alpha", 0.5)
        super(Leapfrog, self).__init__(*args, **kwargs)

    def _call(self, state, timestep):
        """
        Updates the input state dictionary and returns a new state corresponding
        to the next timestep.

        Args
        ----
        state : dict
            The current model state. Will be updated in-place by
            the call due to the Robert-Asselin-Williams filter.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state.
        new_state : dict
            The model state at the next timestep.

        Raises
        ------
        SharedKeyError
            If a DiagnosticComponent object has an output that is
            already in the state at the start of the timestep.
        ValueError
            If the timestep is not the same as the last time
            step() was called on this instance of this object.
        """
        original_state = state
        state = state.copy()
        self._ensure_constant_timestep(timestep)
        tendencies, diagnostics = self.prognostic(state, timestep)
        convert_tendencies_units_for_state(tendencies, state)
        if self._old_state is None:
            new_state = step_forward_euler(state, tendencies, timestep)
        else:
            state, new_state = step_leapfrog(
                self._old_state,
                state,
                tendencies,
                timestep,
                asselin_strength=self._asselin_strength,
                alpha=self._alpha,
            )
        copy_untouched_quantities(state, new_state)
        self._old_state = state
        for key in original_state.keys():
            original_state[key] = state[key]  # allow filtering to be applied
        return diagnostics, new_state

    def _ensure_constant_timestep(self, timestep):
        if self._timestep is None:
            self._timestep = timestep
        elif self._timestep != timestep:
            raise ValueError(
                "timestep must be constant for Leapfrog time stepping"
            )


def step_leapfrog(
    old_state, state, tendencies, timestep, asselin_strength=0.05, alpha=1.0
):
    """
    Steps the model state forward in time using the given tendencies and the
    leapfrog time scheme, with a Robert-Asselin time filter.

    Args
    ----
    old_state : dict
        Model state at the last timestep.
    state : dict
        Model state at the current timestep. May be modified by
        this function call, specifically by the Asselin filter.
    tendencies : dict
        Time derivatives at the current timestep in
        units/second.
    timestep : timedelta
        The amount of time to step forward.
    asselin_strength : float, optional
        Asselin filter strength. Default is 0.05.
    alpha : float, optional
        Constant from Williams (2009), where the
        midpoint is shifted by alpha*influence, and the right point is
        shifted by (alpha-1)*influence. If alpha is 1 then the behavior
        is that of the classic Robert-Asselin time filter, while if it
        is 0.5 the filter will conserve the three-point mean.
        Default is 1.

    Returns
    -------
    state : dict
        The input state, modified in place.
    new_state : dict
        Model state at the next timestep.
    """
    new_state = {}
    for key in tendencies.keys():
        new_state[key] = (
            old_state[key] + 2 * tendencies[key] * timestep.total_seconds()
        )
        filter_influence = (
            0.5
            * asselin_strength
            * (old_state[key] - 2 * state[key] + new_state[key])
        )
        state[key] += alpha * filter_influence
        if alpha != 1.0:
            new_state[key] += (alpha - 1.0) * filter_influence
    return state, new_state


def step_forward_euler(state, tendencies, timestep):
    return_state = {}
    for key in tendencies.keys():
        return_state[key] = (
            state[key] + tendencies[key] * timestep.total_seconds()
        )
    return return_state


def second_bashforth(state, tendencies_list, timestep):
    """Return the new state using second-order Adams-Bashforth. tendencies_list
    should be a list of dictionaries whose values are tendencies in
    units/second (from oldest to newest), and timestep should be a timedelta
    object. The dictionaries in tendencies_list should all have the same keys.
    """
    return_state = {}
    for key in tendencies_list[0].keys():
        return_state[key] = state[key] + timestep.total_seconds() * (
            1.5 * tendencies_list[-1][key] - 0.5 * tendencies_list[-2][key]
        )
    return return_state


def third_bashforth(state, tendencies_list, timestep):
    """Return the new state using third-order Adams-Bashforth. tendencies_list
    should be a list of dictionaries whose values are tendencies in
    units/second (from oldest to newest), and timestep should be a timedelta
    object."""
    return_state = {}
    for key in tendencies_list[0].keys():
        return_state[key] = state[key] + timestep.total_seconds() * (
            23.0 / 12 * tendencies_list[-1][key]
            - 4.0 / 3 * tendencies_list[-2][key]
            + 5.0 / 12 * tendencies_list[-3][key]
        )
    return return_state


def fourth_bashforth(state, tendencies_list, timestep):
    """Return the new state using fourth-order Adams-Bashforth. tendencies_list
    should be a list of dictionaries whose values are tendencies in
    units/second (from oldest to newest), and timestep should be a timedelta
    object."""
    return_state = {}
    for key in tendencies_list[0].keys():
        return_state[key] = state[key] + timestep.total_seconds() * (
            55.0 / 24 * tendencies_list[-1][key]
            - 59.0 / 24 * tendencies_list[-2][key]
            + 37.0 / 24 * tendencies_list[-3][key]
            - 3.0 / 8 * tendencies_list[-4][key]
        )
    return return_state
