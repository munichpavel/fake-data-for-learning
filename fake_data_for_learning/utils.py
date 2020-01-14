'''Utility functions for fake_data_for_learning'''
import numpy as np
import string
from itertools import product

from scipy.special import softmax


class RandomCpt:

    def __init__(self, *shape):
        self.shape = shape

    def __call__(self, seed=None):
        '''
        Generate non-negative random matrix of given 
        shape such that sums over last dimension are 1
        '''
        if seed is not None:
            np.random.seed(seed=seed)
        res = np.random.rand(*self.shape)
        res = self.make_cpt(res)
        return res

    def make_cpt(self, x):
        '''
        Convert numpy array x to conditional
        probability table
        '''
        res = x.copy()
        ranges = [range(s) for s in res.shape[:-1]]
        for s in product(*ranges):
            res[s] = softmax(res[s])

        return res

