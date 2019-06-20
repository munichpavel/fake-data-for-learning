# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np


class BayesianNodeRV:
    '''
    Inspired by rv_discrete of scipy.stats, follows api, except for
    * values here is for the values taken by a RV, not the tuple (xk, pk) of rv_discrete
    '''
    def __init__(self, name, pt, parents=None):
        
        self.name=name
        self.parents = parents
        self.values = self._set_values(pt)
 
    def _set_values(self, pt):
        return np.array(range(pt.size)).reshape(pt.shape)


    def rvs(self, size=None):
        return [self.values[0]]

 