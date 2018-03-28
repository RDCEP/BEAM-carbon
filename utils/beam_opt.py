#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from beam_carbon.beam import BEAMCarbon


class BeamOptimizer(object):
    def __init__(self):
        self.b = BEAMCarbon()
        self.b.time_step = 1.
        self.b.intervals = 12
        a2 = pd.DataFrame.from_csv('../input/a2.csv', index_col=1)
        a2.fillna(0)
        self.b.emissions = np.array(a2.ix[:, 'emissions'][::1])
        self.b.temperature_dependent = False
        self.b.linear_temperature = False

        self.opt_vars = 3
        self.opt_tol = 1e-10  # 1e-5
        self.opt_scale = 1e-5
        self.opt_grad_f = None
        self.opt_obj = None
        self.opt_x = np.zeros(3)

    def f(self, x):
        self.b.delta = x[0]
        self.b.k_d = x[1]
        self.b.land_sink_annual = x[2]

        obj = 0
        # for scenario in ['CLIMBER_RCP6', 'UV_CCN_2', 'UV_CCN_4', 'UV_CCN_6',
        #                  'UV_CCN_8']:
        for scenario in ['UV_CCN_4', 'UV_CCN_6', 'UV_CCN_8']:
            self.b.emissions = np.array(
                pd.DataFrame.from_csv('../src/_opt_emissions.csv').ix[:,
                scenario][6:])
            r = self.b.run()
            y = np.array(r.ix['mass_atmosphere']) / 2.13
            y1 = pd.DataFrame.from_csv('../src/_opt_co2.csv', index_col=0)[256:]
            y1 = np.array(y1[scenario])[::self.b.time_step]
            n = min(len(y), len(y1))
            obj += np.mean((y[:n] - y1[:n]) ** 2) ** .5
        self.opt_obj = obj / 5.
        print(x, self.opt_obj)
        return self.opt_obj

    def grad_f(self, x):
        gf = np.zeros(self.opt_vars)
        y = self.f(x)

        for i in range(len(x)):
            xprime = x.copy()
            xprime[i] += 1e-8
            yprime = self.f(xprime)
            gf[i] = (yprime - y) / 1e-8

        self.opt_grad_f = gf
        return gf

    def optimize(self):
        try:
            import pyipopt
        except ImportError:
            pyipopt = None
            print('OPTIMIZATION ERROR: It appears that you do not have '
                  'pyipopt installed. Please install it before running '
                  'optimization.')
        x0 = np.array([4.9, .0022, 2.5])
        M = 0
        nnzj = 0
        nnzh = 0
        xl = np.array([3, .001, 2.5])
        xu = np.array([50, .05, 2.5])
        gl = np.zeros(M)
        gu = np.ones(M) * 4.0

        def eval_f(_x0):
            if (_x0 == self.opt_x).all() and self.opt_obj is not None:
                return self.opt_obj
            else:
                self.opt_x = _x0.copy()
                return self.f(_x0)

        def eval_grad_f(_x0):
            if (_x0 == self.opt_x).all() and self.opt_grad_f is not None:
                return self.opt_grad_f
            else:
                self.opt_x = _x0.copy()
                return self.grad_f(_x0)

        def eval_g(x):
            return np.zeros(M)

        def eval_jac_g(x, flag):
            if flag:
                return [], []
            else:
                return np.empty(M)

        pyipopt.set_loglevel(2)
        nlp = pyipopt.create(
            self.opt_vars, xl, xu, M, gl, gu, nnzj, nnzh, eval_f,
            eval_grad_f, eval_g, eval_jac_g,
        )
        nlp.num_option('constr_viol_tol', 8e-7)
        nlp.int_option('max_iter', 30000)
        nlp.num_option('max_cpu_time', 60*60)
        nlp.num_option('tol', self.opt_tol)
        # nlp.num_option('obj_scaling_factor', self.opt_scale)
        nlp.str_option('nlp_scaling_method', 'gradient-based')
        nlp.int_option('print_level', 5)
        nlp.str_option('linear_solver', 'ma27')
        x = nlp.solve(x0)[0]
        nlp.close()
        return x

    def slsqp(self):
        from scipy.optimize import minimize
        return minimize(self.grad_f, np.array([.0027]), method='SLSQP',
                        jac=False, bounds=[(.001, .005)])


if __name__ == '__main__':
    b = BeamOptimizer()
    print(b.optimize())
