# -*- coding: utf-8 -*-
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
from sympl._core.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    Monitor,
    Stepper,
    TendencyComponent,
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
from sympl._core.dataarray import DataArray
from sympl._core.exceptions import (
    ComponentExtraOutputError,
    ComponentMissingOutputError,
    DependencyError,
    InvalidPropertyDictError,
    InvalidStateError,
    SharedKeyError,
)
from sympl._core.get_np_arrays import get_numpy_arrays_with_properties
from sympl._core.init_np_arrays import initialize_numpy_arrays_with_properties
from sympl._core.restore_dataarray import restore_data_arrays_with_properties
from sympl._core.tendencystepper import TendencyStepper
from sympl._core.time import datetime, timedelta
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
from sympl._core.util import (
    ensure_no_shared_keys,
    get_component_aliases,
    get_numpy_array,
    jit,
    restore_dimensions,
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

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

__author__ = "Jeremy McGibbon"
__license__ = "BSD"
