# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np


class BayesianNodeRV:
    '''
    Inspired by rv_discrete of scipy.stats, follows api, except for
    * values here is for the values taken by a RV, not the tuple (xk, pk) of rv_discrete
    '''
    def __init__(self, name, pt, values=None, parents=None):
        
        self.name=name
        self.pt = pt
        self.parents = parents
        self.values = self._set_values(pt, values)


    def _set_values(self, pt, values):
        if values is None:
            return np.array(range(pt.size)).reshape(pt.shape)
        else:
            return values


    def rvs(self, size=None, seed=42):
        '''
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.
        '''
        np.random.seed(seed)
        res = np.random.choice(self.values, size, p=self.pt)
        
        return res
 