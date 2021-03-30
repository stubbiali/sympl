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
from sympl._core.dataarray import DataArray
from sympl._core.units import is_valid_unit


class ConstantDict(dict):
    def __repr__(self):
        return self._repr()

    def _repr(self, sphinx=False):
        return_string = ""
        printed_names = set()
        for category, name_list in constant_names_by_category.items():
            if len(name_list) > 0:
                return_string += category.title() + "\n"
            for name in name_list:
                printed_names.add(name)
                units = self[name].attrs["units"]
                units = units.replace("dimensionless", "")
                return_string += "\t{}: {} {}\n".format(
                    name, self[name].values.item(), units
                )
                if sphinx:
                    return_string += "\n"
            return_string += "\n"
        if len(set(self.keys()).difference(printed_names)) > 0:
            return_string += "User Defined\n"
            for name in self.keys():
                if name not in printed_names:
                    units = self[name].attrs["units"]
                    return_string += "\t{}: {} {}\n".format(
                        name, self[name].values.item(), units
                    )
                    if sphinx:
                        return_string += "\n"
            return_string += "\n"
        return return_string

    @property
    def _proxy_dict(self):
        """
        Returns a dictionary which has all possible values one could get out of
        this dictionary (considering aliases).
        """
        return_dict = {}
        return_dict.update(self)
        n_iterations = 0
        while n_iterations < 100:
            new_dict = {}
            for alias, original in constant_aliases.items():
                if (
                    original in return_dict.keys()
                    and alias not in return_dict.keys()
                ):
                    new_dict[alias] = return_dict[original]
            if len(new_dict.keys()) == 0:
                break
            else:
                return_dict.update(new_dict)
        if len(new_dict.keys()) != 0:
            raise RuntimeError("Max number of iterations exceeded.")
        return return_dict

    def keys(self):
        return self._proxy_dict.keys()

    def values(self):
        return self._proxy_dict.values()

    def items(self):
        return self._proxy_dict.items()

    def __setitem__(self, key, value):
        if key in constant_aliases.keys():
            super(ConstantDict, self).__setitem__(get_alias(key), value)
        else:
            super(ConstantDict, self).__setitem__(key, value)

    def __getitem__(self, item):
        if item in constant_aliases.keys():
            return super(ConstantDict, self).__getitem__(get_alias(item))
        else:
            return super(ConstantDict, self).__getitem__(item)


constants = None
constant_aliases = None

default_constant_aliases = {
    "latent_heat_of_condensation": "latent_heat_of_vaporization",
    "enthalpy_of_fusion": "latent_heat_of_fusion",
    "stellar_irradiance": "solar_constant",
    "heat_capacity_of_ice": "heat_capacity_of_solid_phase_as_ice",
    "heat_capacity_of_snow": "heat_capacity_of_solid_phase_as_snow",
    "thermal_conductivity_of_ice": "thermal_conductivity_of_solid_phase_as_ice",
    "thermal_conductivity_of_snow": "thermal_conductivity_of_solid_phase_as_snow",
    "density_of_ice": "density_of_solid_phase_as_ice",
    "density_of_snow": "density_of_solid_phase_as_snow",
}

default_constants = ConstantDict(
    {
        "stefan_boltzmann_constant": DataArray(
            5.670367e-8, attrs={"units": "W m^-2 K^-4"}
        ),
        "gravitational_acceleration": DataArray(
            9.80665, attrs={"units": "m s^-2"}
        ),
        "heat_capacity_of_dry_air_at_constant_pressure": DataArray(
            1004.64, attrs={"units": "J kg^-1 K^-1"}
        ),
        "heat_capacity_of_water_vapor_at_constant_pressure": DataArray(
            1846.0, attrs={"units": "J kg^-1 K^-1"}
        ),
        "specific_enthalpy_of_water_vapor": DataArray(
            2500.0, attrs={"units": "J kg^-1"}
        ),
        "heat_capacity_of_liquid_water": DataArray(
            4185.5, attrs={"units": "J kg^-1 K^-1"}
        ),
        "freezing_temperature_of_liquid_water": DataArray(
            273.0, attrs={"units": "K"}
        ),
        "thermal_conductivity_of_liquid_water": DataArray(
            0.57, attrs={"units": "W m^-1 K^-1"}
        ),
        "heat_capacity_of_solid_water_as_ice": DataArray(
            2108.0, attrs={"units": "J kg^-1 K^-1"}
        ),
        "thermal_conductivity_of_solid_water_as_ice": DataArray(
            2.22, attrs={"units": "W m^-1 K^-1"}
        ),
        "heat_capacity_of_solid_water_as_snow": DataArray(
            2108.0, attrs={"units": "J kg^-1 K^-1"}
        ),
        "thermal_conductivity_of_solid_water_as_snow": DataArray(
            0.2, attrs={"units": "W m^-1 K^-1"}
        ),
        "reference_air_pressure": DataArray(1.0132e5, attrs={"units": "Pa"}),
        "reference_air_temperature": DataArray(300.0, attrs={"units": "degK"}),
        "thermal_conductivity_of_dry_air": DataArray(
            0.026, attrs={"units": "W m^-1 K^-1"}
        ),
        "gas_constant_of_dry_air": DataArray(
            287.0, attrs={"units": "J kg^-1 K^-1"}
        ),
        "gas_constant_of_water_vapor": DataArray(
            461.5, attrs={"units": "J kg^-1 K^-1"}
        ),
        "planetary_rotation_rate": DataArray(
            7.292e-5, attrs={"units": "s^-1"}
        ),
        "planetary_radius": DataArray(6.371e6, attrs={"units": "m"}),
        "latent_heat_of_vaporization_of_water": DataArray(
            2.5e6, attrs={"units": "J kg^-1"}
        ),
        "latent_heat_of_fusion_of_water": DataArray(
            333550.0, attrs={"units": "J kg^-1"}
        ),
        "density_of_liquid_water": DataArray(1e3, attrs={"units": "kg m^-3"}),
        "density_of_solid_water_as_ice": DataArray(
            916.7, attrs={"units": "kg m^-3"}
        ),
        "density_of_solid_water_as_snow": DataArray(
            100.0, attrs={"units": "kg m^-3"}
        ),
        "solar_constant": DataArray(1367.0, attrs={"units": "W m^-2"}),
        "planck_constant": DataArray(6.62607004e-34, attrs={"units": "J s"}),
        "speed_of_light": DataArray(299792458.0, attrs={"units": "m s^-1"}),
        "seconds_per_day": DataArray(
            86400.0, attrs={"units": "dimensionless"}
        ),
        "avogadro_constant": DataArray(
            6.022140857e23, attrs={"units": "mole^-1"}
        ),
        "boltzmann_constant": DataArray(
            1.38064852e-23, attrs={"units": "J K^-1"}
        ),
        "loschmidt_constant": DataArray(2.6516467e25, attrs={"units": "m^-3"}),
        "universal_gas_constant": DataArray(
            8.3144598, attrs={"units": "J mole^-1 K^-1"}
        ),
    }
)

constant_names_by_category = {
    "planetary": [
        "gravitational_acceleration",
        "planetary_radius",
        "planetary_rotation_rate",
        "seconds_per_day",
    ],
    "physical": [
        "stefan_boltzmann_constant",
        "avogadro_constant",
        "speed_of_light",
        "boltzmann_constant",
        "loschmidt_constant",
        "universal_gas_constant",
        "planck_constant",
    ],
    "condensible": [
        "density_of_liquid_phase",
        "heat_capacity_of_liquid_phase",
        "heat_capacity_of_vapor_phase",
        "specific_enthalpy_of_vapor_phase",
        "gas_constant_of_vapor_phase",
        "latent_heat_of_condensation",
        "latent_heat_of_fusion",
        "density_of_solid_phase_as_ice",
        "density_of_solid_phase_as_snow",
        "heat_capacity_of_solid_phase_as_ice",
        "heat_capacity_of_solid_phase_as_snow",
        "thermal_conductivity_of_solid_phase_as_ice",
        "thermal_conductivity_of_solid_phase_as_snow",
        "thermal_conductivity_of_liquid_phase",
        "freezing_temperature_of_liquid_phase",
        "enthalpy_of_fusion",
        "latent_heat_of_vaporization",
    ],
    "atmospheric": [
        "heat_capacity_of_dry_air_at_constant_pressure",
        "gas_constant_of_dry_air",
        "thermal_conductivity_of_dry_air",
        "reference_air_pressure",
        "reference_air_temperature",
    ],
    "stellar": ["stellar_irradiance", "solar_constant",],
    "oceanographic": [],
    "chemical": [
        "heat_capacity_of_water_vapor_at_constant_pressure",
        "density_of_liquid_water",
        "gas_constant_of_water_vapor",
        "latent_heat_of_vaporization_of_water",
        "heat_capacity_of_liquid_water",
        "latent_heat_of_fusion_of_water",
        "heat_capacity_of_solid_water_as_ice",
        "heat_capacity_of_solid_water_as_snow",
        "thermal_conductivity_of_solid_water_as_ice",
        "thermal_conductivity_of_solid_water_as_snow",
        "thermal_conductivity_of_liquid_water",
        "density_of_solid_water_as_ice",
        "density_of_solid_water_as_snow",
        "freezing_temperature_of_liquid_water",
        "specific_enthalpy_of_water_vapor",
        "density_of_snow",
        "heat_capacity_of_snow",
        "heat_capacity_of_ice",
        "density_of_ice",
        "thermal_conductivity_of_ice",
        "thermal_conductivity_of_snow",
    ],
}


def get_condensible_map(condensible_name):
    return {
        "gas_constant_of_vapor_phase": "gas_constant_of_{}_vapor".format(
            condensible_name
        ),
        "heat_capacity_of_vapor_phase": "heat_capacity_of_{}_vapor_at_constant_pressure".format(
            condensible_name
        ),
        "specific_enthalpy_of_vapor_phase": "specific_enthalpy_of_{}_vapor".format(
            condensible_name
        ),
        "density_of_liquid_phase": "density_of_liquid_{}".format(
            condensible_name
        ),
        "heat_capacity_of_liquid_phase": "heat_capacity_of_liquid_{}".format(
            condensible_name
        ),
        "latent_heat_of_vaporization": "latent_heat_of_vaporization_of_{}".format(
            condensible_name
        ),
        "freezing_temperature_of_liquid_phase": "freezing_temperature_of_liquid_{}".format(
            condensible_name
        ),
        "thermal_conductivity_of_liquid_phase": "thermal_conductivity_of_liquid_{}".format(
            condensible_name
        ),
        "latent_heat_of_fusion": "latent_heat_of_fusion_of_{}".format(
            condensible_name
        ),
        "density_of_solid_phase_as_ice": "density_of_solid_{}_as_ice".format(
            condensible_name
        ),
        "density_of_solid_phase_as_snow": "density_of_solid_{}_as_snow".format(
            condensible_name
        ),
        "heat_capacity_of_solid_phase_as_ice": "heat_capacity_of_solid_{}_as_ice".format(
            condensible_name
        ),
        "heat_capacity_of_solid_phase_as_snow": "heat_capacity_of_solid_{}_as_snow".format(
            condensible_name
        ),
        "thermal_conductivity_of_solid_phase_as_ice": "thermal_conductivity_of_solid_{}_as_ice".format(
            condensible_name
        ),
        "thermal_conductivity_of_solid_phase_as_snow": "thermal_conductivity_of_solid_{}_as_snow".format(
            condensible_name
        ),
    }


def get_alias(name):
    n_iterations = 0
    while name in constant_aliases.keys() and n_iterations < 100:
        name = constant_aliases[name]
        n_iterations += 1
    if name in constant_aliases.keys():
        raise RuntimeError(
            "Circular aliases exist for constant name {}. "
            "Max iterations exceeded.".format(name)
        )
    return name


def set_constant(name, value, units):
    """
    Sets the value of a constant.

    Parameters
    ----------
    name : str
        The name of the constant.
    value : float
        The value to which the constant should be set.
    units : str
        The units of the value given.
    """
    if is_valid_unit(units):
        constants[get_alias(name)] = DataArray(value, attrs={"units": units})
    else:
        raise ValueError("{} is not a valid unit.".format(units))


def get_constant(name, units):
    """
    Retrieves the value of a constant.

    Parameters
    ----------
    name : str
        The name of the constant.
    units : str
        The units requested for the returned value.

    Returns
    -------
    value : float
        The value of the constant in the requested units.
    """
    return constants[get_alias(name)].to_units(units).values.item()


def get_constants_string():
    """
    Returns
    -------
    constant_string : str
        A string listing all constants under each category of constants,
        with their current values and units.
    """
    return repr(constants)


def set_condensible_name(name):
    constant_aliases.update(get_condensible_map(name))


def reset_constants():
    """
    Reverts constants to their state when Sympl was originally imported. This
    includes removing any new constants, setting the original constants to
    their original values, and setting the condensible quantity to water.
    """
    global constants
    global constant_aliases
    constants = ConstantDict()
    constants.update(default_constants)
    constant_aliases = {}
    constant_aliases.update(default_constant_aliases)
    constant_aliases.update(get_condensible_map("water"))


reset_constants()


class ConstantList(object):
    __doc__ = constants._repr(sphinx=True)
