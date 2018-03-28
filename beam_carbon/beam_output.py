#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
import os
from six import iteritems
from collections import OrderedDict
from datetime import datetime
import numpy as np
import pandas as pd
from beam_carbon.config import OUTPUT, LOG_ALL_INTERVALS


class BEAMOutput(object):
    """Log output of BEAMCarbon."""
    def __init__(self, beam):
        self.beam = beam
        self._df = None
        self.csv = os.path.join('..', 'output', '{}.csv'.format(
            datetime.now().strftime('%Y%m%d%H%M%S')))

    def __repr__(self):
        return self.df.__repr__()

    @property
    def output_array(self):
        """Map output configuration to BEAMCarbon.

        :return: Dictionary of configuration keys and values
        :rtype: OrderedDict
        """
        return OrderedDict([
            ('mass_atmosphere', self.beam.carbon_mass[0]),
            ('mass_upper', self.beam.carbon_mass[1]),
            ('mass_lower', self.beam.carbon_mass[2]),
            ('temp_atmosphere', self.beam.temperature[0]),
            ('temp_ocean', self.beam.temperature[1]),
            ('cumulative_emissions', self.beam.total_emissions),
            ('phi11', self.beam.transfer_matrix[0][0]),
            ('phi12', self.beam.transfer_matrix[0][1]),
            ('phi13', self.beam.transfer_matrix[0][2]),
            ('phi21', self.beam.transfer_matrix[1][0]),
            ('phi22', self.beam.transfer_matrix[1][1]),
            ('phi23', self.beam.transfer_matrix[1][2]),
            ('phi31', self.beam.transfer_matrix[2][0]),
            ('phi32', self.beam.transfer_matrix[2][1]),
            ('phi33', self.beam.transfer_matrix[2][2]),
            ('A', self.beam.A),
            ('B', self.beam.B),
            ('H', self.beam.H),
            ('k_1', self.beam.k_1),
            ('k_2', self.beam.k_2),
            ('k_h', self.beam.k_h),
            ('pH', -np.log10(self.beam.H)),
        ])

    @property
    def df(self):
        """DataFrame of output values. Default is to log each time step
        ignoring the intervals between. To log every interval, set
        LOG_ALL_INTERVALS = True in config.py.

        :return: Output of model at each time step (or every interval)
        :rtype: pd.DataFrame
        """
        if self._df is None:
            self._df = self.make_initial_df()
        return self._df

    def make_initial_df(self):
        """Create initial DataFrame. Size is based on whether we're
        logging all intervals.

        :return: DataFrame for logging output
        :rtype: pd.DataFrame
        """
        idx = [k for k in self.output_array.keys() if k in OUTPUT]

        if LOG_ALL_INTERVALS:
            n = self.beam.n * self.beam.intervals + 1
            cols = np.arange(n) / self.beam.intervals
        else:
            n = self.beam.n + 1
            cols = np.arange(n) * self.beam.time_step
        return pd.DataFrame(
            np.tile(np.empty(len(idx)).reshape((len(idx), 1)), (n,)),
            index=idx,
            columns=cols, )

    def add_interval(self, i):
        """Add model values at i to DataFrame output.

        :param i: Current interval in model
        :type i: int
        :return: None
        """
        arr = [v for k, v in iteritems(self.output_array) if k in OUTPUT]
        self.df.iloc[:, i] = np.array(arr)

    def to_csv(self):
        """Write output to a csv file.

        :return: None
        """
        self.df.to_csv(self.csv)
