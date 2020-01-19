'''Utility functions for fake_data_for_learning'''
import string
from itertools import product

import numpy as np
import pandas as pd

from scipy.special import softmax


class RandomCpt:
    """Generate random conditional probability table"""
    def __init__(self, *shape, linear_constraints=None):
        self.shape = shape

    def __call__(self, seed=None):
        """
        Generate non-negative random matrix of given 
        shape such that sums over last dimension are 1
        """
        if seed is not None:
            np.random.seed(seed=seed)
        res = np.random.rand(*self.shape)
        res = self.make_cpt(res)
        return res

    @staticmethod
    def make_cpt(x):
        """
        Convert numpy array x to conditional
        probability table
        """
        res = x.copy()
        ranges = [range(s) for s in res.shape[:-1]]
        for s in product(*ranges):
            res[s] = softmax(res[s])

        return res


class MultidimIndexToLinearMapping:
    """Convert multidimensional array indices to linear and back"""
    def __init__(self, *multidim_index_values):
        self.multidim_index_values = multidim_index_values
        self.mapping = self._set_mapping(multidim_index_values)

    def _set_mapping(self, multidim_index_values):
        multidim_values = list(product(*multidim_index_values))
        return pd.Series(range(len(multidim_values)), index=multidim_values)

    def to_linear(self, multi_index_value):
        return self.mapping[multi_index_value]

    def to_multidim(self, linear_index_value):
        return self.mapping.index[linear_index_value]