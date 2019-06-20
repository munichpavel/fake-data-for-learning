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
            return np.array(range(pt.shape[0]))
        else:
            return values

    
    def rvs(self, parent_values=None, size=None, seed=42):
        '''
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.
        '''
        np.random.seed(seed)

        if self.parents is None:
            return np.random.choice(self.values, size, p=self.pt)
        else:
            res = np.random.choice(self.values, size, p=self.get_pt(parent_values))
            return res
    

    def get_pt(self, parent_values=None):
        if parent_values is None:
            return self.pt
        else:
            s = [slice(None)] * len(self.pt.shape)
            for idx, p in enumerate(self.parents):
                s[idx + 1] = parent_values[p]
            return self.pt[tuple(s)]
 