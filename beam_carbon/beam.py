#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from collections import OrderedDict
import os
from math import floor
from datetime import datetime
from six import iteritems
import numpy as np
import pandas as pd
from config import OUTPUT
from beam_carbon.temperature import DICETemperature, LinearTemperature


__version__ = '0.3'


class BEAMCarbon(object):
    """Class for computing BEAM carbon cycle from emissions input.
    """
    def __init__(self, emissions=None, time_step=1, intervals=100):
        """BEAMCarbon init

        Args:
            :param emissions: Array of annual emissions in GtC, beginning
                in 2005
            :type emissions: list
            :param time_step: Time between emissions values in years
            :type time_step: float
            :param intervals: Number of times to calculate BEAM carbon
                per timestep
            :type intervals: int
        """
        self._temperature_dependent = True
        self._intervals = intervals
        self._time_step = time_step
        self.temperature_mod = DICETemperature(self.time_step, self.intervals, 0)
        self._temperature = self.temperature_mod.initial_temp.copy()
        self.total_emissions = 0

        if emissions is not None and type(emissions) in [list, np.ndarray]:
            self._emissions = np.array(emissions)
        else:
            self._emissions = np.zeros(1)

        self._k_1 = 8e-7
        self._k_2 = 4.53e-10
        self._k_h = 1.23e3
        self._k_d = .05
        self._A = None
        self._B = None
        self._H = None
        self._Alk = 767.
        self._delta = 50.
        self._initial_carbon = np.array([808.9, 725., 35641.])
        self._carbon_mass = None
        self._linear_temperature = False
        self.csv = os.path.join('..', 'output', '{}.csv'.format(
            datetime.now().strftime('%Y%m%d%H%M%S')))
        self.log_all_output = False

    @property
    def initial_carbon(self):
        """Values for initial carbon in atmosphere, upper and lower oceans
        in GtC. Default values are from 2005.

        :return: Three layer carbon mass at timestep 0
        :rtype: np.ndarray
        """
        return self._initial_carbon

    @initial_carbon.setter
    def initial_carbon(self, value):

        value = np.array(value, dtype=np.float)

        if len(value) != 3:
            raise ValueError(
                'BEAMCarbon.initial_carbon must have three values.')

        if np.any(np.isnan(value)):
            raise TypeError(
                'BEAMCarbon.initial_carbon must have three non-negative '
                'values.')

        if len(np.where(value < 0)[0]) > 0:
            raise TypeError(
                'BEAMCarbon.initial_carbon must have three non-negative '
                'values.')

        self._initial_carbon = value

    @property
    def carbon_mass(self):
        """Values for carbon in atmosphere, upper and lower oceans at each
        timestep in GtC. Default values for time 0 are equal to initial_carbon.

        :return: Three layer carbon mass
        :rtype: np.ndarray
        """
        if self._carbon_mass is None:
            self._carbon_mass = self.initial_carbon.copy()
        return self._carbon_mass

    @carbon_mass.setter
    def carbon_mass(self, value):
        self._carbon_mass = np.array(value, dtype=np.float)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        value = np.array(value, dtype=np.float)

        if len(value) != 2:
            raise ValueError(
                'BEAMCarbon.temperature must have two numeric values.')

        if np.any(np.isnan(value)):
            raise TypeError(
                'BEAMCarbon.temperature must have two numeric values.')

        self._temperature = value

    @property
    def transfer_matrix(self):
        """3 by 3 matrix of transfer coefficients for carbon cycle.

        :return: Transfer matrix
        :rtype: np.ndarray
        """
        return np.array([
            -self.k_a, self.k_a * self.A * self.B, 0,
            self.k_a, -(self.k_a * self.A * self.B) - self.k_d,
            self.k_d / self.delta,
            0, self.k_d, -self.k_d / self.delta,
        ]).reshape((3, 3,))

    @property
    def emissions(self):
        """Array of annual emissions values in GtC per timestep. Note that
        no matter what size the timesteps are, the emissions values should be
        annual.

        :return: Emissions values
        :rtype: np.ndarray
        """
        return self._emissions

    @emissions.setter
    def emissions(self, value):
        value = np.array(value, dtype=np.float)

        if np.any(np.isnan(value)):
            raise TypeError(
                'BEAMCarbon.emissions must contain numeric values.')

        self._emissions = value
        self.temperature_mod.n = self.n

    @property
    def time_step(self):
        """Size of time steps in emissions array.

        :return: Size of timestep
        :rtype: float
        """
        return self._time_step

    @time_step.setter
    def time_step(self, value):
        if type(value) not in [float, int] or value <= 0:
            raise TypeError(
                'BEAMCarbon.time_step must be a positive numeric value.')
        self._time_step = value
        self.temperature_mod.time_step = self.time_step

    @property
    def n(self):
        """Number of timesteps.

        :return: Number of timesteps
        :rtype: int
        """
        return len(self.emissions)

    @property
    def intervals(self):
        """Number of intervals in each time step.
        """
        return self._intervals

    @intervals.setter
    def intervals(self, value):
        if type(value) not in [int] or value <= 0:
            raise TypeError(
                'BEAMCarbon.intervals must be a positive integer value.')

        self._intervals = value
        self.temperature_mod.periods = self.intervals

    @property
    def k_a(self):
        """Constant k_{a}

        :return: k_{a}
        :rtype: float
        """
        return .2

    @property
    def k_d(self):
        """Constant k_{d}

        :return: k_{d}
        :rtype: float
        """
        return self._k_d

    @k_d.setter
    def k_d(self, value):
        self._k_d = value

    @property
    def delta(self):
        """Ratio of lower ocean to upper ocean.

        :return: Ratio
        :rtype: float
        """
        return self._delta

    @delta.setter
    def delta(self, value):
        self.initial_carbon[1] = 725 * 51 / (value + 1)
        self.initial_carbon[2] = 36366 - self.initial_carbon[1]
        self.Alk = 767 * 51 / (value + 1)
        self._delta = value

    @property
    def k_h(self):
        """CO2 solubility.

        :return: k_{h}
        :rtype: float
        """
        if self._k_h is None:
            self._k_h = self.get_kh(self.temperature_mod.initial_temp[1])
        return self._k_h

    @k_h.setter
    def k_h(self, value):
        self._k_h = value

    @property
    def k_1(self):
        """First dissociation constant.

        :return: k_{1}
        :rtype: float
        """
        if self._k_1 is None:
            self._k_1 = self.get_k1(self.temperature_mod.initial_temp[1])
        return self._k_1

    @k_1.setter
    def k_1(self, value):
        self._k_1 = value

    @property
    def k_2(self):
        """Second dissociation constant.

        :return: k_{2}
        :rtype: float
        """
        if self._k_2 is None:
            self._k_2 = self.get_k2(self.temperature_mod.initial_temp[1])
        return self._k_2

    @k_2.setter
    def k_2(self, value):
        self._k_2 = value

    @property
    def AM(self):
        """Moles in the atmosphere.

        :return: AM
        :rtype: float
        """
        return 1.77e20

    @property
    def OM(self):
        """Moles in the ocean.

        :return: OM
        :rtype: float
        """
        return 7.8e22

    @property
    def Alk(self):
        """Alkalinity in GtC.

        :return: Alkalinity
        :rtype: float
        """
        return self._Alk

    @Alk.setter
    def Alk(self, value):
        self._Alk = value

    @property
    def A(self):
        """Ratio of mass of CO2 in atmospheric to upper ocean dissolved CO2.

        :return: A
        :rtype: float
        """
        if self._A is None:
            if self.temperature_dependent:
                self.k_h = self.get_kh(self.temperature_mod.initial_temp[1])
            self._A = self.get_A()
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def B(self):
        """Ratio of dissolved CO2 to total oceanic carbon.

        :return: B
        :rtype: float
        """
        if self._B is None:
            self.temp_calibrate(self.temperature_mod.initial_temp[1])
            self._B = self.get_B(self.H)
        return self._B

    @B.setter
    def B(self, value):
        self._B = value

    @property
    def H(self):
        if self._H is None:
            self._H = self.get_H(self.carbon_mass[1])
        return self._H

    @H.setter
    def H(self, value):
        self._H = value

    @property
    def salinity(self):
        """Salinity in g / kg of seawater.

        :return: Salinity
        :rtype: float
        """
        return 35.

    @property
    def temperature_dependent(self):
        """Switch for calculating temperature-dependent parameters k_{1},
        k_{2}, and k_{h}.

        :return: Temperature dependence state
        :rtype: bool
        """
        return self._temperature_dependent

    @property
    def linear_temperature(self):
        """Switch for calculating linear temperature rather than DICE.

        :return: Linear temperature state
        :rtype: bool
        """
        return self._linear_temperature

    @temperature_dependent.setter
    def temperature_dependent(self, value):
        if type(value) is bool:
            self._temperature_dependent = value
        else:
            raise TypeError(
                'BEAMCarbon.temperature_dependent must be True or False.')

    @linear_temperature.setter
    def linear_temperature(self, value):
        if value:
            self.temperature_mod = LinearTemperature(self.time_step,
                                                     self.intervals, self.n)
        else:
            self.temperature_mod = DICETemperature(self.time_step,
                                                   self.intervals, self.n)
        self._linear_temperature = value

    def temp_calibrate(self, to):
        """Recalibrate temperature-dependent parameters k_{1}, k_{2}, and k_{h}.

        :param to: ocean temperature (C)
        :type to: float
        :return: None
        :rtype: None
        """
        self.k_1 = self.get_k1(to)
        self.k_2 = self.get_k2(to)
        self.k_h = self.get_kh(to)
        self.A = self.get_A()

    def get_B(self, h):
        """Calculate B (Ratio of dissolved CO2 to total oceanic carbon),
         given H (the concentration of hydrogen ions)

        :param h: H, concentration of hydrogen ions [H+] (the (pH) of seawater)
        :type h: float
        :return: B, ratio of dissolved CO2 to total oceanic carbon
        :rtype: float
        """
        return 1 / (1 + self.k_1 / h + self.k_1 * self.k_2 / h ** 2)

    def get_A(self):
        """Calculate A based on temperature-dependent changes in k_{h}

        :return: A
        :rtype: float
        """
        return self.k_h * self.AM / (self.OM / (self.delta + 1))

    def get_H(self, mass_upper):
        """Solve for [H+], the concentration of hydrogen ions
        (the (pH) of seawater).

        :param mass_upper: Carbon mass in oceans in GtC
        :type mass_upper: float
        :return: H
        :rtype: float
        """
        p0 = 1
        p1 = (self.k_1 - mass_upper * self.k_1 / self.Alk)
        p2 = (1 - 2 * mass_upper / self.Alk) * self.k_1 * self.k_2
        return max(np.roots([p0, p1, p2]))

    def get_kh(self, temp_ocean):
        """Calculate temperature dependent k_{h}

        :param temp_ocean: change in upper ocean temperature (C)
        :type temp_ocean: float
        :return: k_h
        :rtype: float
        """
        t = 283.15 + temp_ocean
        k0 = np.exp(
            9345.17 / t - 60.2409 + 23.3585 * np.log(t / 100.) +
            self.salinity * (
                .023517 - .00023656 * t + .0047036 * (t / 100.) ** 2))
        kh = 1 / (k0 * 1.027) * 55.57
        self.A = kh * self.AM / (self.OM / (self.delta + 1.))
        return kh

    def get_pk1(self, t):
        """Calculate pk1, exponent of k_{1}.
         k_{1} = 10 ** -pk1

        :param t: change in upper ocean temperature (C)
        :type t: float
        :return: pk1
        :rtype: float
        """
        return (
            -13.721 + 0.031334 * t + 3235.76 / t + 1.3e-5 * self.salinity * t -
            0.1031 * self.salinity ** 0.5)

    def get_pk2(self, t):
        """Calculate pk1, exponent of k_{1}.
        k_{2} = 10 ** -pk2

        :param t: change in upper ocean temperature (C)
        :type t: float
        :return: pk2
        :rtype: float
        """
        return (
            5371.96 + 1.671221 * t + 0.22913 * self.salinity +
            18.3802 * np.log10(self.salinity)) - (128375.28 / t +
            2194.30 * np.log10(t) + 8.0944e-4 * self.salinity * t +
            5617.11 * np.log10(self.salinity) / t) + 2.136 * self.salinity / t

    def get_k1(self, temp_ocean):
        """Calculate temperature dependent k_{1}

        :param temp_ocean: change in upper ocean temperature (C)
        :type temp_ocean: float
        :return: k_{1}
        :rtype: float
        """
        return 10 ** -self.get_pk1(283.15 + temp_ocean)

    def get_k2(self, temp_ocean):
        """Calculate temperature dependent k_{2}

        :param temp_ocean: change in upper ocean temperature (C)
        :type temp_ocean: float
        :return: k_{2}
        :rtype: float
        """
        return 10 ** -self.get_pk2(283.15 + temp_ocean)

    def add_output(self, i=None, output=None):

        darr = OrderedDict([
            ('mass_atmosphere', self.carbon_mass[0]),
            ('mass_upper', self.carbon_mass[1]),
            ('mass_lower', self.carbon_mass[2]),
            ('temp_atmosphere', self.temperature[0]),
            ('temp_ocean', self.temperature[1]),
            ('cumulative_emissions', self.total_emissions),
            ('phi11', self.transfer_matrix[0][0]),
            ('phi12', self.transfer_matrix[0][1]),
            ('phi13', self.transfer_matrix[0][2]),
            ('phi21', self.transfer_matrix[1][0]),
            ('phi22', self.transfer_matrix[1][1]),
            ('phi23', self.transfer_matrix[1][2]),
            ('phi31', self.transfer_matrix[2][0]),
            ('phi32', self.transfer_matrix[2][1]),
            ('phi33', self.transfer_matrix[2][2]),
            ('A', self.A),
            ('B', self.B),
            ('H', self.H),
            ('k_1', self.k_1),
            ('k_2', self.k_2),
            ('k_h', self.k_h),
        ])

        arr = []
        idx = []

        for k, v in iteritems(darr):
            if k in OUTPUT and i is None:
                idx.append(k)
                arr.append(v)
            elif k in OUTPUT:
                arr.append(v)

        if output is None or i is None:
            return pd.DataFrame(
                np.tile(np.array(arr).reshape((len(arr), 1)), (self.n + 1,)),
                index=idx,
                columns=np.arange(self.n + 1) * self.time_step,)

        output.iloc[:, i] = np.array(arr)

        return output

    def land_sink(self, carbon_mass, i):
        annual_sink = 2.5
        years_of_sink = 300
        return carbon_mass - (
            annual_sink * self.time_step / self.intervals) * (
            (years_of_sink * self.intervals - self.time_step * self.intervals) /
            (years_of_sink * self.intervals))

    def log(self, temp_atmosphere, temp_ocean, total_carbon, h, i):
        if i == 0:
            with open(self.csv, 'w') as f:
                f.write('Ma,Mu,Ml,Ta,To,ka*A*B,TC,A,B,kh,H,pH\n')
        with open(self.csv, 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                *np.concatenate((
                    (self.carbon_mass.copy() - self.initial_carbon) / 2.13,
                    np.array([temp_atmosphere, temp_ocean]),
                    np.array([self.transfer_matrix[0][1],
                              total_carbon,
                              self.A, self.B, self.k_h, h,
                              -np.log10(h)]))).tolist()))

    def run(self):
        """Run the BEAM model.

        :return: DataFrame of values over the entire run
        :rtype: pd.DataFrame
        """
        N = self.n * self.intervals
        self.carbon_mass = self.initial_carbon.copy()
        emissions = np.zeros(3)
        output = self.add_output()

        for i in range(N):

            _i = int(floor(i / self.intervals)) # time_step

            if i % self.intervals == 0 and self.temperature_dependent:
                self.temp_calibrate(self.temperature[1])

            self.H = self.get_H(self.carbon_mass[1])
            self.B = self.get_B(self.H)

            emissions[0] = self.emissions[_i] * self.time_step / self.intervals
            self.total_emissions += emissions[0]

            self.carbon_mass += (
                np.multiply((self.transfer_matrix * self.carbon_mass),
                            self.time_step / self.intervals).sum(axis=1) +
                emissions)
            self.carbon_mass[0] = self.land_sink(self.carbon_mass[0], i)

            if (i + 1) % self.intervals == 0:

                ta = self.temperature[0]
                self.temperature[0] = self.temperature_mod.temp_atmosphere(
                    index=_i, temp_atmosphere=ta,
                    temp_ocean=self.temperature[1],
                    mass_atmosphere=self.carbon_mass[0],
                    carbon=self.total_emissions,
                    initial_carbon=self.initial_carbon,
                    phi11=self.transfer_matrix[0][0],
                    phi21=self.transfer_matrix[1][0])
                self.temperature[1] = self.temperature_mod.temp_ocean(
                    ta, self.temperature[1])

                output = self.add_output(_i+1, output)

        self.A = None
        self.B = None
        self.H = None
        self.carbon_mass = None

        return output


def main():
    def create_args():
        import argparse
        parser = argparse.ArgumentParser(
            description='BEAM carbon cycle on the command line.'
        )
        input_group = parser.add_mutually_exclusive_group()
        input_group.add_argument(
            '-e', '--emissions', type=str,
            help='Comma separated values of annual emissions.')
        input_group.add_argument(
            '-c', '--input', '--csv', type=str,
            help='Path to CSV file to use as input.')
        parser.add_argument(
            '-t', '--timestep', type=float, default=1,
            help='Time step for input values in years. Default is 1.')
        parser.add_argument(
            '-i', '--intervals', type=int, default=10,
            help='BEAM calculation intervals per time step. Default is 100.')
        parser.add_argument(
            '-o', '--output', type=str, default='beam_output.csv',
            help='Write values to CSV file instead of stdout')
        parser.add_argument(
            '-T', '--tempdependent', type=bool, default=False,
            help='Recalibrate k_h, k_1, and k_2 based on temperature of '
                 'upper ocean at each interval.')

        return parser.parse_args()

    args = create_args()

    def run_beam(e):
        beam = BEAMCarbon(e)
        if args.timestep:
            beam.time_step = args.timestep
        if args.intervals:
            beam.intervals = args.intervals
        if args.tempdependent:
            beam.temperature_dependent = True
        return beam.run()

    def write_beam(output, csv=None):
        if csv is not None:
            output.to_csv(csv)
        else:
            print(output.to_string())
        return True

    csv = args.output if args.output else None
    emissions = np.array([float(n) for n in args.emissions.split(',')]) \
        if args.emissions else None

    if args.input:
        with open(args.input, 'r') as f:
            for line in f:
                write_beam(run_beam(line.split(',')), csv=csv)
    else:
        write_beam(run_beam(emissions), csv=csv)


if __name__ == '__main__':
    ############################################################################
    # The following is intended merely as an example of how to run the         #
    # BEAM code in a python interpreter. It should be modified to suit         #
    # the needs of the user.                                                   #
    ############################################################################
    b = BEAMCarbon()                        # Create a BEAMCarbon object.
    b.time_step = 1.                        # 1 year times steps.
    b.intervals = 24                        # Run BEAM 24 times each time step.
    a2 = pd.DataFrame.from_csv(             # Load emissions input from CSV.
        os.path.join(
            '..', 'input', 'a2.csv', index_col=1))
    a2.fillna(0)
    b.emissions = np.array(                 # Set emissions property with array
        a2.ix[:, 'emissions'])              # from CSV.
    b.delta = 5                             # Change the default delta.
    b.k_d = .002                            # Change the default k_{d}.
    b.temperature_dependent = False         # Don't recalculate k_{h}.
    b.linear_temperature = False            # Use DICE temperature model.
    print(b.run())                          # Run the model & print the output.
