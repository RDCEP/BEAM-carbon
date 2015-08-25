#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import floor
import numpy as np


class Temperature(object):
    def __init__(self, time_step, periods, n):
        self.time_step = time_step
        self.periods = periods
        self.n = n

    @property
    def initial_temp(self):
        return np.array([.7307, .0068])

    @property
    def forcing_ghg_2000(self):
        """Value from DICE2010
        """
        return .83

    @property
    def forcing_ghg_2100(self):
        return .30

    @property
    def transfer_matrix(self):
        """Values from DICE2010
        """
        return np.array([.208, 0., .310, .050])

    @property
    def forcing_co2_doubling(self):
        return 3.8

    @property
    def temp_co2_doubling(self):
        return 3.2

    @property
    def mass_pi(self):
        return 592.14

    @property
    def forcing_ghg(self):
        """Forcing equation

        F_EX, Exogenous forcing for other greenhouse gases

        Returns:
            nd.array: Array of forcing values, n=params.tmax

        """
        _n = int(floor(100 / self.time_step))
        a = self.forcing_ghg_2000 + (1 / _n) * (
                self.forcing_ghg_2100 - self.forcing_ghg_2000
            ) * np.arange(_n + 1)
        b = self.forcing_ghg_2100 * np.ones(self.n - (_n + 1)) if self.n > _n else np.array([])
        return np.concatenate((a, b))[:self.n]

    def forcing(self, index, mass_atmosphere):
        """Forcing equation

        F, Forcing, W/m^2

        Returns:
            float: Forcing

        """
        return (
            self.forcing_co2_doubling *
            (np.log(
                mass_atmosphere / self.mass_pi
            ) / np.log(2)) + self.forcing_ghg[index]
        )


class DICETemperature(Temperature):

    def temp_atmosphere(self, index, temp_atmosphere, temp_lower, mass_atmosphere=None):
        """T_AT, increase in atmospheric temperature since 1750, degrees C

         Args:
            :param temp_atmosphere: Atmospheric temperature at t-1
             :type temp_atmosphere: float
            :param temp_lower: Lower ocean temperature at t-1
             :type temp_lower: float
            :param forcing: Forcings at t
             :type forcing: float

        Returns:
            :returns: T_AT(t-1) + ξ_1 * (F(t) - F2xCO2 / T2xCO2 * T_AT(t-1) - ξ_3 * (T_AT(t-1) - T_Ocean(t-1)))
              :rtype: float
        """
        return (
            temp_atmosphere +
            self.transfer_matrix[0] * (
                self.forcing(index, mass_atmosphere) - (self.forcing_co2_doubling /
                           self.temp_co2_doubling) *
                temp_atmosphere - self.transfer_matrix[2] *
                (temp_atmosphere - temp_lower)
            )
        )

    def temp_lower(self, temp_atmosphere, temp_lower):
        """T_LO, increase in atmospheric temperature since 1750, degrees C

         Args:
            :param i: current time step
             :type i: int
            :param df: Matrix of variable values
             :type df: DiceDataMatrix

        Returns:
            :returns: T_Ocean(t-1) + ξ_4 * (T_AT(t-1) - T_Ocean(t-1))
            :rtype: float
        """
        return (
            temp_lower + self.transfer_matrix[3] *
            (temp_atmosphere - temp_lower)
        )

class LinearTemp(Temperature):
    def temp_atmosphere(self, index, temp_atmosphere, temp_lower, mass_atmosphere=None):
        #TODO: 1.7 * cumulative carbon emitted - pre-industrial mass
        pass