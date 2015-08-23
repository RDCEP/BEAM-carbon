#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import sqrt
import numpy as np
from sympy import symbols, solve, Eq


class BEAMCarbon(object):
    def __init__(self, emissions=None, time_step=1, periods=10):
        self._emissions = emissions if emissions is not None else np.zeros(100)
        self._periods = periods
        self._time_step = time_step
        self._A = None
        self._B = None
        self._transfer_matrix = np.array([
            -self.k_a, self.k_a * self.A * self.B, 0,
            self.k_a, -(self.k_a * self.A * self.B) - self.k_d, self.k_d / self.delta,
            0, self.k_d, -self.k_d / self.delta,
        ]).reshape((3, 3,))

    @property
    def initial_carbon(self):
        return np.array([808.9, 725., 35641.])

    @property
    def transfer_matrix(self):
        return self._transfer_matrix

    # @transfer_matrix.setter
    # def transfer_matrix(self, value):
    #     self._transfer_matrix = value

    @property
    def emissions(self):
        return self._emissions

    @emissions.setter
    def emissions(self, value):
        self._emissions = value

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, value):
        self._time_step = value

    @property
    def n(self):
        return len(self.emissions)

    @property
    def periods(self):
        return self._periods

    @periods.setter
    def periods(self, value):
        self._periods = value

    @property
    def k_a(self):
        return .2

    @property
    def k_d(self):
        return .05

    @property
    def delta(self):
        return 50.

    @property
    def k_h(self):
        # return 1.91e3
        return 1.23e3

    @property
    def k_1(self):
        return 8e-7

    @property
    def k_2(self):
        return 4.53e-10

    @property
    def AM(self):
        return 1.77e20

    @property
    def OM(self):
        return 7.8e22

    @property
    def Alk(self):
        return 767.

    @property
    def A(self):
        if self._A is None:
            #TODO Get initial temps
            self.get_A(.7307)
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def B(self):
        if self._B is None:
            #TODO Get initial temps
            self._B = self.get_B(self.get_h(self.initial_carbon[1]))
        return self._B

    @B.setter
    def B(self, value):
        self._B = value

    @property
    def salinity(self):
        return 35.

    def run(self):
        output = np.tile(np.concatenate((
            self.initial_carbon,
            self.transfer_matrix.reshape((9)))).reshape((12, 1)).copy(),
            self.n + 1)
        mass_tmp = self.initial_carbon.copy()
        N = self.periods * self.n

        for y in xrange(self.n):
            for p in xrange(self.periods):

                h = self.get_h(mass_tmp[1])
                self.B = self.get_B(h)
                mass_tmp += ((
                    self.transfer_matrix * mass_tmp +
                    np.array([self.emissions[y] * self.time_step, 0, 0])) / self.periods).sum(axis=1)

            output[:, y+1] = np.concatenate((mass_tmp.copy(), self.transfer_matrix.reshape((9))))
        return output

    def get_B(self, h):
        return 1 / (1 + self.k_1 / h + self.k_1 * self.k_2 / h ** 2)

    def get_A(self, t):
        self.get_kh(t)

    def get_h(self, mu):
        h = symbols('h')
        a = mu / self.Alk
        f = Eq(
            (h**2 + self.k_1 * h + self.k_1 * self.k_2) / self.k_1,
            a * (h + 2 * self.k_2)
        )
        return max(solve(f, h))

    def get_h_alt(self):
        h, MA = symbols('h, MA')
        f = Eq((1 + self.k_1 / h + self.k_1 * self.k_2 / h ** 2) * self.Alk,
               (self.k_1 / h + 2 * self.k_1 * self.k_2 / h ** 2) * MA)

    def get_kh(self, t):
        """

        :param t: temperature (C)
        :return:
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
        """

        :param t: temperature (C)
        :return:
        """
        t += 273.15
        pk1 = (
            -13.721 + 0.031334 * t + 3235.76 / t + 1.3e-5 * self.salinity * t -
            0.1031 * self.salinity ** 0.5)
        return 10 ** -pk1

    def get_k2(self, t):
        """

        :param t: temperature (C)
        :return:
        """
        t += 273.15
        pk2 = (
            5371.96 + 1.671221 * t + 0.22913 * self.salinity +
            18.3802 * np.log(self.salinity) - 128375.28 / t -
            2194.30 * np.log(t) - 8.0944e-4 * self.salinity * t -
            5617.11 * np.log(self.salinity) + 2.136 * self.salinity)
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
            '-i', '--input', type=str,
            help='Path to CSV file to use as input')
        parser.add_argument(
            '-t', '--timestep', type=float, default=1,
            help='Time step for input values in years. Default is 1.')
        parser.add_argument(
            '-p', '--periods', type=int, default=10,
            help='BEAM calculation periods per time step. Default is 10.')
        parser.add_argument(
            '-o', '--output', action='store_true', default=False,
            help='Write values to CSV file instead of stdout')

        return parser.parse_args()

    args = create_args()

    if args.emissions:
        emissions = np.array([float(n) for n in args.emissions.split(',')])
    else:
        emissions = [10.,13.]

    def run_beam(beam):
        if args.timestep:
            beam.time_step = args.timestep
        if args.periods:
            beam.periods = args.periods
        return beam.run()

    def write_beam(output, csv=None):

        o = ''
        for row in output:
            # print(row)
            o += ','.join([str(r) for r in row]) + '\n'
        if csv is not None:
            with open(csv, 'w') as f:
                f.write(o)
        print(o)
        return True

    if args.input:
        with open(args.input, 'r') as f:
            for line in f:
                beam_carbon = BEAMCarbon(line.split(','))
                write_beam(run_beam(beam_carbon))
    else:
        beam_carbon = BEAMCarbon(emissions)
        write_beam(run_beam(beam_carbon))


if __name__ == '__main__':
    b = BEAMCarbon()
    b.time_step = 10.
    b.periods = 20
    b.emissions = [10,13]
    print b.run()