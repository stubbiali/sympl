# -*- coding: utf-8 -*-
from sympl._core.base_components import (
    TendencyComponent,
    DiagnosticComponent,
    Stepper,
    Monitor,
    ImplicitTendencyComponent,
)
from sympl._core.composite import (
    TendencyComponentComposite,
    DiagnosticComponentComposite,
    MonitorComposite,
    ImplicitTendencyComponentComposite,
)
from sympl._core.tendencystepper import TendencyStepper
from sympl._components.timesteppers import (
    AdamsBashforth,
    Leapfrog,
    SSPRungeKutta,
)
from sympl._core.exceptions import (
    InvalidStateError,
    SharedKeyError,
    DependencyError,
    InvalidPropertyDictError,
    ComponentExtraOutputError,
    ComponentMissingOutputError,
)
from sympl._core.dataarray import DataArray
from sympl._core.constants import (
    get_constant,
    set_constant,
    set_condensible_name,
    reset_constants,
    get_constants_string,
)
from sympl._core.tracers import (
    register_tracer,
    get_tracer_unit_dict,
    get_tracer_input_properties,
    get_tracer_names,
)
from sympl._core.util import (
    ensure_no_shared_keys,
    get_numpy_array,
    jit,
    restore_dimensions,
    get_component_aliases,
)
from sympl._core.combine_properties import combine_component_properties
from sympl._core.units import (
    units_are_same,
    units_are_compatible,
    is_valid_unit,
)
from sympl._core.get_np_arrays import get_numpy_arrays_with_properties
from sympl._core.restore_dataarray import restore_data_arrays_with_properties
from sympl._core.init_np_arrays import initialize_numpy_arrays_with_properties
from sympl._components import (
    PlotFunctionMonitor,
    NetCDFMonitor,
    RestartMonitor,
    ConstantTendencyComponent,
    ConstantDiagnosticComponent,
    RelaxationTendencyComponent,
    TimeDifferencingWrapper,
)
from sympl._core.wrappers import UpdateFrequencyWrapper, ScalingWrapper
from sympl._core.time import datetime, timedelta

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
    InvalidStateError,
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
    ensure_no_shared_keys,
    get_numpy_array,
    jit,
    register_tracer,
    get_tracer_unit_dict,
    get_tracer_input_properties,
    get_tracer_names,
    restore_dimensions,
    get_numpy_arrays_with_properties,
    restore_data_arrays_with_properties,
    initialize_numpy_arrays_with_properties,
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
    timedelta,
)

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

__author__ = "Jeremy McGibbon"
__license__ = "BSD"
