# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np


class BayesianNodeRV:
    '''
    Sample-able random variable corresponding to node of a discrete Bayesian network.
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
 