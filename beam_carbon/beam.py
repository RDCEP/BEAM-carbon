#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import floor
import numpy as np
from sympy import symbols, solve, Eq
from beam_carbon.temperature import DICETemperature


class BEAMCarbon(object):
    """Class for computing BEAM carbon cycle from emissions input.
    """
    def __init__(self, emissions=None, time_step=1, intervals=10):
        """

        :param emissions: Array of annual emissions in GtC, beginning in 2005
        :type emissions: list
        :param time_step: Time between emissions values in years
        :type time_step: float
        :param intervals: Nuber of times to calculate BEAM carbon per timestep
        :type intervals: int
        :param e0: Emissions at t0 if emissions in each timestep are unknown
                   (incompatible with emissions parameter)
        :type e0: float
        :return: None
        """
        self._temperature_dependent = True
        self._intervals = intervals
        self._time_step = time_step
        self.temperature = DICETemperature(self.time_step, self.intervals, 0)

        if emissions is not None and type(emissions) in [list, np.ndarray]:
            self.emissions = emissions
        else:
            self.emissions = np.zeros(100)

        self._k_1 = 8e-7
        self._k_2 = 4.53e-10
        self._k_h = 1.23e3
        self._A = None
        self._B = None
        self._transfer_matrix = np.array([
            -self.k_a, self.k_a * self.A * self.B, 0,
            self.k_a, -(self.k_a * self.A * self.B) - self.k_d,
            self.k_d / self.delta,
            0, self.k_d, -self.k_d / self.delta,
        ]).reshape((3, 3,))

    @property
    def initial_carbon(self):
        """Values for initial carbon in atmosphere, upper and lower oceans
        in GtC. Default values are from 2005.
        """
        return np.array([808.9, 725., 35641.])

    @property
    def transfer_matrix(self):
        """3 by 3 matrix of transfer coefficients for carbon cycle.
        """
        # return self._transfer_matrix
        return np.array([
            -self.k_a, self.k_a * self.A * self.B, 0,
            self.k_a, -(self.k_a * self.A * self.B) - self.k_d,
            self.k_d / self.delta,
            0, self.k_d, -self.k_d / self.delta,
        ]).reshape((3, 3,))

    @property
    def emissions(self):
        """Array of emissions values in GtC per year.
        """
        return self._emissions

    @emissions.setter
    def emissions(self, value):
        self._emissions = value
        self.temperature.n = self.n

    @property
    def time_step(self):
        """Size of time steps in emissions array.
        """
        return self._time_step

    @time_step.setter
    def time_step(self, value):
        self._time_step = value
        self.temperature.time_step = self.time_step

    @property
    def n(self):
        return len(self.emissions)

    @property
    def intervals(self):
        """Number of intervals in each time step.
        """
        return self._intervals

    @intervals.setter
    def intervals(self, value):
        self._intervals = value
        self.temperature.periods = self.intervals

    @property
    def k_a(self):
        """Time constant k_{a} (used for building transfer matrix).
        """
        return .2

    @property
    def k_d(self):
        """Time constant k_{d} (used for building transfer matrix).
        """
        return .05

    @property
    def delta(self):
        """Ratio of lower ocean to upper ocean (used for building transfer
        matrix).
        """
        return 50.

    @property
    def k_h(self):
        """CO2 solubility.
        """
        return self._k_h

    @k_h.setter
    def k_h(self, value):
        self._k_h = value

    @property
    def k_1(self):
        """First dissacoiation constant.
        """
        return self._k_1

    @k_1.setter
    def k_1(self, value):
        self._k_1 = value

    @property
    def k_2(self):
        """Second dissacoiation constant.
        """
        return self._k_2

    @k_2.setter
    def k_2(self, value):
        self._k_2 = value

    @property
    def AM(self):
        """Moles in the atmosphere.
        """
        return 1.77e20

    @property
    def OM(self):
        """Moles in the ocean.
        """
        return 7.8e22

    @property
    def Alk(self):
        """Alkalinity in GtC.
        """
        return 767.

    @property
    def A(self):
        """Ratio of mass of CO2 in atmospheric to upper ocean dissolved CO2.
        """
        if self._A is None:
            if self.temperature_dependent:
                self.k_h = self.get_kh(self.temperature.initial_temp[1])
            self.get_A()
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def B(self):
        """Ratio of dissolved CO2 to total oceanic carbon.
        """
        if self._B is None:
            self.temp_calibrate(self.temperature.initial_temp[1])
            self._B = self.get_B(self.get_h(self.initial_carbon[1]))
        return self._B

    @B.setter
    def B(self, value):
        self._B = value

    @property
    def salinity(self):
        """Salinity in g / kg of seawater.
        """
        return 35.

    @property
    def temperature_dependent(self):
        """Switch for calculating temperature-dependent paramters k_1,
        k_2, and k_h.
        """
        return self._temperature_dependent

    @temperature_dependent.setter
    def temperature_dependent(self, value):
        if type(value) is bool:
            self._temperature_dependent = value
        else:
            raise TypeError('BEAMCarbon.temperature_dependent must be True or False.')

    def temp_calibrate(self, to):
        """Recalibrate temperature-dependent parameters k_1, k_2, and k_h.
        """
        self.k_1 = self.get_k1(to)
        self.k_2 = self.get_k2(to)
        self.k_h = self.get_kh(to)
        self.A = self.get_A()

    def run(self):
        output = np.tile(np.concatenate((
            self.initial_carbon,
            self.temperature.initial_temp,
            self.transfer_matrix.reshape(9))).reshape((14, 1)).copy(),
            self.n + 1)
        mass_tmp = self.initial_carbon.copy()
        emissions = np.zeros(3)
        ta, to = self.temperature.initial_temp

        for i in xrange(self.n * self.intervals):

            _i = int(floor(i / self.intervals)) # time_step
            if self.temperature_dependent:
                self.temp_calibrate(to)
            h = self.get_h(mass_tmp[1])
            self.B = self.get_B(h)

            if i % self.intervals == 0:

                emissions[0] = self.emissions[_i] * self.time_step
                ta = self.temperature.temp_atmosphere(_i, ta, to, mass_tmp[0])
                to = self.temperature.temp_lower(ta, to)

            mass_tmp += ((self.transfer_matrix * mass_tmp + emissions) /
                         self.intervals).sum(axis=1)

            if (i + 1) % self.intervals == 0:
                output[:, _i + 1] = (
                    np.concatenate((mass_tmp.copy(), np.array([ta, to]),
                                    self.transfer_matrix.reshape((9)))))
        return output

    def get_B(self, h):
        """Calculate B given H

        :param h: H, concentration of hydrogen ions [H+] (the (pH) of seawater)
        :type h: float
        :return: B, ratio of dissolved CO2 to total oceanic carbon
        :rtype: float
        """
        return 1 / (1 + self.k_1 / h + self.k_1 * self.k_2 / h ** 2)

    def get_A(self):
        """Calculate A based on temperature-dependent changes in k_h

        :return: A
        :rtype: float
        """
        self.k_h * self.AM / (self.OM / (self.delta + 1))

    def get_h(self, mu):
        """Solve for H+, the concentration of hydrogen ions [H+]
        (the (pH) of seawater).

        :param t: Carbon mass in ocenas in GtC
        :type t: float
        :return: H
        :rtype: float
        """
        h = symbols('h')
        a = mu / self.Alk
        f = Eq(
            (h**2 + self.k_1 * h + self.k_1 * self.k_2) / self.k_1,
            a * (h + 2 * self.k_2)
        )
        return max(solve(f, h))

    def get_kh(self, t):
        """Calculate temperature dependent k_h

        :param t: temperature (C)
        :type t: float
        :return: k_h
        :rtype: float
        """
        t += 283.15
        k0 = np.exp(
            9345.17 / t - 60.2409 + 23.3585 * np.log(t / 100.) +
            self.salinity * (
                .023517 - .00023656 * t + .0047036 * (t / 100) ** 2))
        kh = 1 / (k0 * 1.027) * 55.57
        self.A = kh * self.AM / (self.OM / (self.delta + 1.))
        return kh

    def get_k1(self, t):
        """Calculate temperature dependent k_1

        :param t: temperature (C)
        :type t: float
        :return: k_1
        :rtype: float
        """
        t += 283.15
        pk1 = (
            -13.721 + 0.031334 * t + 3235.76 / t + 1.3e-5 * self.salinity * t -
            0.1031 * self.salinity ** 0.5)
        return 10 ** -pk1

    def get_k2(self, t):
        """Calculate temperature dependent k_2

        :param t: temperature (C)
        :type t: float
        :return: k_2
        :rtype: float
        """
        t += 283.15
        pk2 = (
            5371.96 + 1.671221 * t + 0.22913 * self.salinity +
            18.3802 * np.log10(self.salinity)) - (128375.28 / t +
            2194.30 * np.log10(t) + 8.0944e-4 * self.salinity * t +
            5617.11 * np.log10(self.salinity) / t) + 2.136 * self.salinity / t
        return 10 ** -pk2


def main():
    def create_args():
        import argparse
        parser = argparse.ArgumentParser(
            description='TKTK.'
        )
        input_group = parser.add_mutually_exclusive_group()
        input_group.add_argument(
            '-e', '--emissions', type=str,
            help='Comma separated values to use as emissions input.')
        input_group.add_argument(
            '-c', '--input', '--csv', type=str,
            help='Path to CSV file to use as input.')
        unknown_emissions = input_group.add_argument(
            '-f', '--emissions0', type=float,
            help='Annual emissions in initial time period.'
        )
        parser.add_argument(
            '-p', '--periods', type=int,
            help='Periods to calculate in emissions path.'
        )
        parser.add_argument(
            '-g', '--growth', type=float,
            help='Denominator of emissions growth.'
        )
        parser.add_argument(
            '-t', '--timestep', type=float, default=1,
            help='Time step for input values in years. Default is 1.')
        parser.add_argument(
            '-i', '--intervals', type=int, default=10,
            help='BEAM calculation intervals per time step. Default is 10.')
        parser.add_argument(
            '-o', '--output', action='store_true', default=False,
            help='Write values to CSV file instead of stdout')

        return parser.parse_args()

    args = create_args()

    def run_beam(e):
        beam = BEAMCarbon(e)
        if args.timestep:
            beam.time_step = args.timestep
        if args.intervals:
            beam.intervals = args.intervals
        if args.emissions0:
            beam.emissions = args.emissions0 * np.exp(-np.arange(args.periods)/args.growth)
        return beam.run()

    def write_beam(output, csv=None):
        o = ''
        for row in output:
            o += ','.join([str(r) for r in row]) + '\n'
        if csv is not None:
            with open(csv, 'w') as f:
                f.write(o)
        else:
            print(o)
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
    # b = BEAMCarbon()
    # b.time_step = 10.
    # b.intervals = 20
    # b.emissions = [10,13,1]
    # print b.run()
    main()