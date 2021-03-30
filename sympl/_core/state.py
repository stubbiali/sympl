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
def copy_untouched_quantities(old_state, new_state):
    for key in old_state.keys():
        if key not in new_state:
            new_state[key] = old_state[key]


def add(state_1, state_2):
    out_state = {}
    if "time" in state_1.keys():
        out_state["time"] = state_1["time"]
    for key in state_1.keys():
        if key != "time":
            out_state[key] = state_1[key] + state_2[key]
            if hasattr(out_state[key], "attrs"):
                out_state[key].attrs = state_1[key].attrs
    return out_state


def multiply(scalar, state):
    out_state = {}
    if "time" in state.keys():
        out_state["time"] = state["time"]
    for key in state.keys():
        if key != "time":
            out_state[key] = scalar * state[key]
            if hasattr(out_state[key], "attrs"):
                out_state[key].attrs = state[key].attrs
    return out_state
