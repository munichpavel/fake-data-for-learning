# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np

from scipy.stats._distn_infrastructure import rv_sample


class BayesianNodeRV(rv_sample):
    '''
    See https://github.com/scipy/scipy/issues/8057 for why simple
    subclassing from rv_discrete does not work
    '''
    def __init__(self, name, pt, *args, parents=None, **kwargs):
        
        self.name=name
        self.parents = parents
        self._xk = self._set_xk(pt)
        super().__init__(self, values=(self._xk, pt), *args, **kwargs)

 
    def _set_xk(self, pt):
        return np.array(range(pt.size)).reshape(pt.shape)

 