# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np

from scipy.stats._distn_infrastructure import rv_sample

class BayesianNodeRV(rv_sample):
    '''
    See https://github.com/scipy/scipy/issues/8057 for why simple
    subclassing from rv_discrete does not work
    '''
    def __new__(cls, *args, **kwds):
        return super(BayesianNodeRV, cls).__new__(cls)

    def __init__(self, name, pt, input_vars=None):
        super()
        self.name=name
        self.pk = pt
        self.xk = self._set_xk(pt)
        self.input_vars = input_vars
        self.pmf = self._set_pmf(name, pt, input_vars)
        

    def _set_xk(self, pt):
        return np.array(range(pt.size)).reshape(pt.shape)


    def _set_pmf(self, name, pt, input_vars):
        def pmf(k, input_vals):
            if k == 0:
                return 0.2
            elif k == 1:
                return 0.8
        return pmf