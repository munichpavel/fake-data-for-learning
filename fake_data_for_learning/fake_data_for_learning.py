# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np


class BayesianNodeRV:
    '''
    Inspired by ```rv_discrete``` of scipy.stats
    '''
    def __init__(self, name, pt, parents=None):
        
        self.name=name
        self.parents = parents
        self._xk = self._set_xk(pt)
 
    def _set_xk(self, pt):
        return np.array(range(pt.size)).reshape(pt.shape)

 