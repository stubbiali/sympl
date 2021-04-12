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
from sympl._components import (
    ConstantDiagnosticComponent,
    ConstantTendencyComponent,
    NetCDFMonitor,
    PlotFunctionMonitor,
    RelaxationTendencyComponent,
    RestartMonitor,
    TimeDifferencingWrapper,
)
from sympl._components.timesteppers import (
    AdamsBashforth,
    Leapfrog,
    SSPRungeKutta,
)
from sympl._core.core_components import (
    TendencyComponent,
    ImplicitTendencyComponent,
    DiagnosticComponent,
)
from sympl._core.combine_properties import combine_component_properties
from sympl._core.composite import (
    DiagnosticComponentComposite,
    ImplicitTendencyComponentComposite,
    MonitorComposite,
    TendencyComponentComposite,
)
from sympl._core.constants import (
    get_constant,
    get_constants_string,
    reset_constants,
    set_condensible_name,
    set_constant,
)
from sympl._core.data_array import DataArray
from sympl._core.exceptions import (
    ComponentExtraOutputError,
    ComponentMissingOutputError,
    DependencyError,
    InvalidPropertyDictError,
    InvalidDataArrayDictError,
    SharedKeyError,
)
from sympl._core.monitors import Monitor
from sympl._core.steppers import TendencyStepper, Stepper
from sympl._core.time import datetime
from sympl._core.tracers import (
    get_tracer_input_properties,
    get_tracer_names,
    get_tracer_unit_dict,
    register_tracer,
)
from sympl._core.units import (
    is_valid_unit,
    units_are_compatible,
    units_are_same,
)
from sympl._core.utils import (
    get_component_aliases,
    jit,
)
from sympl._core.wrappers import ScalingWrapper, UpdateFrequencyWrapper

__all__ = (
    TendencyComponent,
    DiagnosticComponent,
    Stepper,
    Monitor,
    TendencyComponentComposite,
    ImplicitTendencyComponentComposite,
    DiagnosticComponentComposite,
    MonitorComposite,
    ImplicitTendencyComponent,
    TendencyStepper,
    Leapfrog,
    AdamsBashforth,
    SSPRungeKutta,
    InvalidDataArrayDictError,
    SharedKeyError,
    DependencyError,
    InvalidPropertyDictError,
    ComponentExtraOutputError,
    ComponentMissingOutputError,
    units_are_same,
    units_are_compatible,
    is_valid_unit,
    DataArray,
    get_constant,
    set_constant,
    set_condensible_name,
    reset_constants,
    get_constants_string,
    TimeDifferencingWrapper,
    jit,
    register_tracer,
    get_tracer_unit_dict,
    get_tracer_input_properties,
    get_tracer_names,
    get_component_aliases,
    combine_component_properties,
    PlotFunctionMonitor,
    NetCDFMonitor,
    RestartMonitor,
    ConstantTendencyComponent,
    ConstantDiagnosticComponent,
    RelaxationTendencyComponent,
    UpdateFrequencyWrapper,
    ScalingWrapper,
    datetime,
)

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

__author__ = "Jeremy McGibbon"
__license__ = "BSD"
