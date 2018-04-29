#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import os
from math import floor
import numpy as np
import pandas as pd
from beam_carbon.temperature import DICETemperature, LinearTemperature
from beam_carbon.beam_output import BEAMOutput


__version__ = '0.3.2'


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
        self.intervals = intervals
        self.time_step = time_step
        self.temperature_mod = DICETemperature(
            self.time_step, self.intervals, 0)
        self.total_emissions = 0

        if emissions is not None and type(emissions) in [list, np.ndarray]:
            self.emissions = np.array(emissions)
        else:
            self.emissions = np.zeros(1)

        self.n = len(self.emissions)
        self.k_1 = np.empty(self.n * self.intervals)
        self.k_2 = np.empty(self.n * self.intervals)
        self.k_h = np.empty(self.n * self.intervals)
        self.pk1 = np.empty(self.n * self.intervals)
        self.pk2 = np.empty(self.n * self.intervals)
        self.H = np.empty(self.n * self.intervals)
        self.A = np.empty(self.n * self.intervals)
        self.B = np.empty(self.n * self.intervals)
        self.mass_atmosphere = np.empty(self.n * self.intervals)
        self.mass_upper = np.empty(self.n * self.intervals)
        self.mass_lower = np.empty(self.n * self.intervals)
        self.temp_atmosphere = np.empty(self.n * self.intervals)
        self.temp_ocean = np.empty(self.n * self.intervals)
        self.k_1[:] = 8e-7
        self.k_2[:] = 4.53e-10
        self.k_h[:] = 1.23e3
        self.k_d = .05
        self.k_a = .2
        self.AM = 1.77e20
        self.OM = 7.8e22
        self.Alk = 767.
        self.delta = 50.
        self.mass_atmosphere[0] = 808.9
        self.mass_upper[0] = 725.
        self.mass_lower[0] = 35641.
        self.temp_atmosphere[0] = .7307
        self.temp_ocean[0] = .0068
        self.salinity = 35.
        self.annual_land_sink = 2.5
        self.land_sink_years = 300
        self.linear_temperature = False
        self.temperature_dependent = True
        self.A[:] = self.get_A(0)
        self.H[:] = self.get_H(0)
        self.B[:] = self.get_B(0)
        self.output = BEAMOutput(self)
        if self.linear_temperature:
            self.temperature_mod = LinearTemperature(self.time_step,
                                                     self.intervals, self.n)
        else:
            self.temperature_mod = DICETemperature(self.time_step,
                                                   self.intervals, self.n)

    def transfer_matrix(self, i):
        """3 by 3 matrix of transfer coefficients for carbon cycle.

        :return: Transfer matrix
        :rtype: np.ndarray
        """
        return np.array([
            -self.k_a, self.k_a * self.A[i] * self.B[i], 0,
            self.k_a, -(self.k_a * self.A[i] * self.B[i]) - self.k_d,
            self.k_d / self.delta,
            0, self.k_d, -self.k_d / self.delta,
        ]).reshape((3, 3,))

    def temp_calibrate(self, i):
        """Recalibrate temperature-dependent parameters k_{1}, k_{2}, and k_{h}.

        :return: None
        :rtype: None
        """
        self.pk1[i] = self.get_pk1(i)
        self.k_1[i] = self.get_k1(i)
        self.pk2[i] = self.get_pk2(i)
        self.k_2[i] = self.get_k2(i)
        self.k_h[i] = self.get_kh(i)
        self.A[i] = self.get_A(i)

    def get_B(self, i):
        """Calculate B (Ratio of dissolved CO2 to total oceanic carbon),
         given H (the concentration of hydrogen ions)

        :param h: H, concentration of hydrogen ions [H+] (the (pH) of seawater)
        :type h: float
        :return: B, ratio of dissolved CO2 to total oceanic carbon
        :rtype: float
        """
        return 1 / (1 + self.k_1[i] / self.H[i] +
                    self.k_1[i] * self.k_2[i] / self.H[i] ** 2)

    def get_A(self, i):
        """Calculate A based on temperature-dependent changes in k_{h}

        :return: A
        :rtype: float
        """
        return self.k_h[i] * self.AM / (self.OM / (self.delta + 1))

    def get_H(self, i):
        """Solve for [H+], the concentration of hydrogen ions
        (the (pH) of seawater).

        :return: H
        :rtype: float
        """
        j = i if i == 0 else i - 1
        return (0.5 * (np.sqrt(self.k_1[i]) * np.sqrt(
            self.k_1[i] * (self.mass_upper[j] / self.Alk) ** 2 -
            2 * self.k_1[i] * self.mass_upper[j] / self.Alk + self.k_1[i] +
            8 * self.k_2[i] * self.mass_upper[j] / self.Alk -
            4 * self.k_2[i]) + self.k_1[i] * self.mass_upper[j] / self.Alk -
                       self.k_1[i]))

    def get_kh(self, i):
        """Calculate temperature dependent k_{h}
        """
        t = 283.15 + self.temp_ocean[i]
        k0 = np.exp(
            9345.17 / t - 60.2409 + 23.3585 * np.log(t / 100.) +
            self.salinity * (
                .023517 - .00023656 * t + .0047036 * (t / 100.) ** 2))
        kh = 1 / (k0 * 1.027) * 55.57
        return kh

    def get_pk1(self, i):
        """Calculate pk1, exponent of k_{1}.
         k_{1} = 10 ** -pk1

        :return: pk1
        :rtype: float
        """
        t = (283.15 + self.temp_ocean[i])
        return (
            -13.721 + 0.031334 * t + 3235.76 / t + 1.3e-5 * self.salinity * t -
            0.1031 * self.salinity ** 0.5)

    def get_pk2(self, i):
        """Calculate pk1, exponent of k_{1}.
        k_{2} = 10 ** -pk2

        :return: pk2
        :rtype: float
        """
        t = (283.15 + self.temp_ocean[i])
        return (
            5371.96 + 1.671221 * t + 0.22913 * self.salinity +
            18.3802 * np.log10(self.salinity)) - (
            128375.28 / t + 2194.30 * np.log10(t) +
            8.0944e-4 * self.salinity * t +
            5617.11 * np.log10(self.salinity) / t) + 2.136 * self.salinity / t

    def get_k1(self, i):
        """Calculate temperature dependent k_{1}

        :return: k_{1}
        :rtype: float
        """
        return 10 ** -self.pk1[i]

    def get_k2(self, i):
        """Calculate temperature dependent k_{2}

        :return: k_{2}
        :rtype: float
        """
        return 10 ** -self.pk2[i]

    def write_csv(self):
        import os
        from datetime import datetime
        csv = os.path.join('..', 'output', '{}.csv'.format(
            datetime.now().strftime('%Y%m%d%H%M%S')))
        with open(csv, 'w') as f:

            f.write('k1,{}\n'.format(','.join(map(str, self.k_1))))
            f.write('k2,{}\n'.format(','.join(map(str, self.k_2))))
            f.write('kh,{}\n'.format(','.join(map(str, self.k_h))))
            f.write('A,{}\n'.format(','.join(map(str, self.A))))
            f.write('B,{}\n'.format(','.join(map(str, self.B))))
            f.write('H,{}\n'.format(','.join(map(str, self.H))))
            f.write('pk1,{}\n'.format(','.join(map(str, self.pk1))))
            f.write('pk2,{}\n'.format(','.join(map(str, self.pk2))))
            f.write('Ta,{}\n'.format(','.join(map(str, self.temp_atmosphere))))
            f.write('To,{}\n'.format(','.join(map(str, self.temp_ocean))))
            f.write('Ma,{}\n'.format(','.join(map(str, self.mass_atmosphere))))
            f.write('Mu,{}\n'.format(','.join(map(str, self.mass_upper))))
            f.write('Ml,{}\n'.format(','.join(map(str, self.mass_lower))))

    def run(self):
        """Run the BEAM model.

        :return: DataFrame of values over the entire run
        :rtype: pd.DataFrame
        """
        N = self.n * self.intervals
        emissions = np.zeros(3)

        for i in range(1, N):

            _i = int(floor(i / self.intervals))

            self.temp_atmosphere[i] = self.temperature_mod.temp_atmosphere(
                index=_i, temp_atmosphere=self.temp_atmosphere[i - 1],
                temp_ocean=self.temp_ocean[i - 1],
                mass_atmosphere=self.mass_atmosphere[i - 1],
            )
            self.temp_ocean[i] = self.temperature_mod.temp_ocean(
                self.temp_atmosphere[i - 1], self.temp_ocean[i - 1])

            if self.temperature_dependent:
                self.temp_calibrate(i)

            self.H[i] = self.get_H(i)
            self.B[i] = self.get_B(i)
            self.A[i] = self.get_A(i)

            emissions[0] = self.emissions[_i] * self.time_step / self.intervals
            self.total_emissions += emissions[0]
            delta_carbon = np.multiply(
                (self.transfer_matrix(i) * np.array([self.mass_atmosphere[i-1],
                                                  self.mass_upper[i-1],
                                                  self.mass_lower[i-1]])),
                 self.time_step / self.intervals).sum(axis=1) + emissions
            self.mass_atmosphere[i] = self.mass_atmosphere[i-1] + delta_carbon[0]
            self.mass_upper[i] = self.mass_upper[i-1] + delta_carbon[1]
            self.mass_lower[i] = self.mass_lower[i-1] + delta_carbon[2]


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
            '-d', '--delta', type=float, default='50',
            help='Value for delta (ratio of lower ocean to upper)')
        parser.add_argument(
            '-k', '--kd', type=float, default='.05',
            help='Transfer coefficient from upper to lower ocean')
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
        if args.delta:
            beam.delta = args.delta
        if args.kd:
            beam.k_d = args.kd
        return beam.run()

    def write_beam(output, csv=None):
        if csv is not None:
            output.to_csv(csv)
        else:
            print(output.to_string())
        return True

    csv_file = args.output if args.output else None
    emissions = np.array([float(n) for n in args.emissions.split(',')]) \
        if args.emissions else None

    if args.input:
        with open(args.input, 'r') as f:
            for line in f:
                write_beam(run_beam(line.split(',')), csv=csv_file)
    else:
        write_beam(run_beam(emissions), csv=csv_file)


if __name__ == '__main__':
    ############################################################################
    # The following is intended merely as an example of how to run the         #
    # BEAM code in a python interpreter. It should be modified to suit         #
    # the needs of the user.                                                   #
    ############################################################################
    a2 = pd.DataFrame.from_csv(             # Load emissions input from CSV.
        os.path.join(
            '..', 'input', 'a2.csv'), index_col=1)
    a2.fillna(0)
    b = BEAMCarbon(
        np.array(a2.ix[:, 'emissions']),
        1.,
        120,
    )
    b.temperature_dependent = True
    b.linear_temperature = False
    b.run()
    b.write_csv()
