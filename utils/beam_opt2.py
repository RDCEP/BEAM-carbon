#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
from math import ceil
import pandas as pd
import numpy as np
from lmfit import minimize, Parameters
from beam_carbon.beam import BEAMCarbon
import matplotlib.pyplot as plt
from matplotlib import rc


class BeamOptimizer(object):
    def __init__(self):
        pass

    def opt_ocean(self, params):
        return self.compare(initial_carbon=[808.9,
                                            params[3]*1e2,
                                            params[4]*1e4])

    def opt_k(self, params):
        return self.compare(delta=params[0], k_d=params[1]*1e-2,
                            k_a=params[2]*1e-1, )

    def opt_all(self, params):
        return self.compare(delta=params[0], k_d=params[1]*1e-2,
                            k_a=params[2]*1e-1,
                            initial_carbon=[808.9,
                                            params[3]*1e2,
                                            params[4]*1e4])

    def compare(self, **kwargs):
        font = {'family': 'sans',
                'weight': 'bold',
                'size': 8}
        rc('font', **font)
        fig = plt.figure(figsize=(8.5, 11), dpi=80)

        error = 0

        scenarios = [
            # ['uv', 'ccl6', 300],
            # ['uv', 'ccl6', None],
            ['uv', 'ccn6', 300],
            ['uv', 'ccn6', None],
            # ['uv', 'cl6', 300],
            # ['uv', 'cl6', None],
            ['uv', 'a2p', 300],
            ['uv', 'a2p', None],
            ['uv', 'ccn8', 300],
            ['uv', 'ccn8', None],
            # ['uv', 'cl6', 300],
            # ['uv', 'cl6', None],
        ]

        for i, scenario in enumerate(scenarios):
            n = 0
            if scenario[0] == 'c2':
                n = 6
            co2 = pd.DataFrame.from_csv(
                '../input/{}_output.csv'.format(scenario[0]),
                index_col=0, header=0)['{}_{}'.format(*scenario)][n:]
            emit = pd.DataFrame.from_csv(
                '../input/{}_scenarios.csv'.format(scenario[0]),
                index_col=0, header=0)[scenario[1]][n:]
            emit = np.array(emit)[:min(len(co2), len(emit))]
            co2 = np.array(co2)[:min(len(co2), len(emit))]
            if scenario[2] is not None:
                co2 = co2[:scenario[2]]

            # Add emissions for 2005
            emit = np.concatenate((np.array([6.7]), emit))

            beam = BEAMCarbon([])
            beam.time_step = 1
            beam.intervals = 12
            beam.emissions = emit
            for k, v in kwargs.iteritems():
                beam.__setattr__(k, v)
            beam.annual_land_sink = 0.
            # if scenario[0] == 'c2':
            #     beam.annual_land_sink = 2.5
            # else:
            #     beam.annual_land_sink = 3.
            r = beam.run()
            if scenario[2] is not None:
                y = np.array(r.ix['mass_atmosphere'])[1:scenario[2]+1:beam.time_step]
            else:
                y = np.array(r.ix['mass_atmosphere'])[1:-1:beam.time_step]
            # y /= 2.13
            # print(d, kd, ka, np.mean((y - co2) ** 2) ** .5)
            err = np.mean((y - co2) ** 2) ** .5
            error += err

            ax = fig.add_subplot(len(scenarios), 1, i+1)
            ax.set_title(
                '{}_{} {} (RMS error={})'.format(
                    scenario[0], scenario[1],
                    ' '.join(
                        ['{}={}'.format(k, round(v, 4))
                         for k, v in kwargs.iteritems()]),
                    round(err, 2)))

            foo = y[1:] + (y[0] - y[1])
            beam_plt, = ax.plot(np.arange(len(y)) + 2005,
                                np.concatenate((y[:1], foo)), label='BEAM')
            uvic_plt, = ax.plot(np.arange(len(y)) + 2005, co2[:],
                                label=scenario)
            ax.set_ylim([0, 5000])
            ax.legend(handles=[beam_plt, uvic_plt])

        print(kwargs, error)
        plt.tight_layout()
        plt.savefig('../output/BEAM_{}.pdf'.format(datetime.now()))
        plt.close()
        return error

    def optimize_lbfgsb(self):
        from scipy.optimize import fmin_l_bfgs_b
        _opt = fmin_l_bfgs_b(self.opt_k,
                             # np.array([50., 5., 2.]),
                             np.array([2., .1, .5]),
                             bounds=[[1., 80.], [0.01, 10.], [0., 5.], ],
                             pgtol=1e-1, factr=1e5,
                             approx_grad=True,
                             epsilon=1e-4,
                             iprint=3
                             )
        return _opt

    def optimize(self):
        # return self.optimize_lm()
        return self.optimize_lbfgsb()
        # return self.optimize_curvefit()


class BeamBasinhopperTakeStep(object):
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize

    def __call__(self, x):
        """

        :param x: [delta, k_d, k_a]
        :type x: np.ndarray
        :return:
        :rtype:
        """
        s = self.stepsize
        # 50, .05, .50
        steps = np.array([50, .05, .5])
        step = np.random.uniform(np.ones(3) * 1e-8, steps) - x
        x += step
        print('x:', x, 'step:', step)
        return x


class BeamStagedOptimizer(object):
    def __init__(self, scenario, stage_length):
        self.scenario = scenario
        self.stage_length = stage_length
        self.df = pd.DataFrame.from_csv(
            '../input/{}.csv'.format(self.scenario),
            index_col=0, header=0)[256:]
        self.steps = len(self.df['EMIT'])
        self.stages = ceil(self.steps / self.stage_length)

    def optimize(self):
        ds = np.empty(self.steps)
        ds[:] = 50
        kds = np.empty(self.steps)
        kds[:] = .05
        kas = np.empty(self.steps)
        kas[:] = .2
        for i in range(int(self.stages)):
            a = i * self.stage_length
            b = min(a + self.stage_length, self.steps)
            optimizer = BeamOptimizer(self.scenario, df=self.df,
                                      b=b, ds=ds, kds=kds, kas=kas)
            optimal = optimizer.optimize()
            ds[a:b] = optimal.params['d'].value
            kds[a:b] = optimal.params['kd'].value
            kas[a:b] = optimal.params['ka'].value
        return ds, kds, kas


if __name__ == '__main__':
    from beam_carbon.beam import BEAMCarbon

    o = BeamOptimizer()
    opt = o.optimize()
    print(opt)

# delta, kd, ka
# First 300
# array([  3.98632156e+00,   2.10176479e-03,   4.99999992e-01])

# [['uv', 'ccl6'], ['uv', 'ccn6'], ['uv', 'cl6'], ]
# Land sink 3
# (array([ 15.34560154,   0.91243647,   4.39611479]), 204.01173120564658,
# {'warnflag': 0, 'task': 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', 'grad': array([ -2.47955138e+01,   4.38012498e+02,   1.39803445e-01]), 'nit': 10, 'funcalls': 136})
# Land sink 0
# (array([ 1.00255828,  0.1       ,  0.47029489]), 73.582204651792495,
# {'warnflag': 0, 'task': 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL', 'grad': array([ -2.70306850e-02,   3.39949696e+02,   9.13574027e-03]), 'nit': 25, 'funcalls': 196})
# First 300 years, land sink 3
# (array([ 14.01786365,   1.14945375,   5.        ]), 80.449989194958548,
# {'warnflag': 0, 'task': 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', 'grad': array([ -9.92871046e+00,   1.23196904e+02,   6.98109564e-02]), 'nit': 17, 'funcalls': 168})
# First 300 years, land sink 0
# (array([ 2.03407108,  0.341533  ,  3.8427052 ]), 27.052029109080149,
# {'warnflag': 0, 'task': 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', 'grad': array([ 0.4544905 , -0.33944794,  1.10645996]), 'nit': 16, 'funcalls': 180})

# ['uv', 'ccl6', 300], ['uv', 'ccl6', None], ['uv', 'ccn6', 300], ['uv', 'ccn6', None], ['uv', 'cl6', 300], ['uv', 'cl6', None],
# Land sink 0
# (array([ 1.00247559,  0.1       ,  0.45780757]), 123.11392730573205,
# {'warnflag': 0, 'task': 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL', 'grad': array([ -6.79653681e-03,   2.35379216e+01,  -2.43930359e-03]), 'nit': 30, 'funcalls': 316})

# ['uv', 'ccn6', 300], ['uv', 'ccn6', None], ['uv', 'ccn8', 300], ['uv', 'ccn8', None], ['uv', 'a2p', 300], ['uv', 'a2p', None],
# Land sink 0
# (array([ 2.50704896,  0.1488002 ,  5.        ]), 529.77393286376889,
# {'warnflag': 0, 'task': 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', 'grad': array([  0.57867551,  12.4673997 ,  -0.35555462]), 'nit': 24, 'funcalls': 220})
