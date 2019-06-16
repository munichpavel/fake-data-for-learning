# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np
from itertools import product

class GraphProbabilityTable:
    r'''
    Probability table for a node in a Bayesian network for
    a discrete random variable X = (X_0, ..., X_{m-1}) admitting
    a graph decomposition G
    '''
    
    def __init__(self, pt, I, J, K, r):
        self.pt = pt
        self.I = I
        self.J  = J
        self.K = K
        self.r = r


    def embed_pt(self):
        r'''
        Embed the probability table of a vertex of (X, G) into the ambient space of the 
        probability table of X, i.e. in R^{r_0} x ... x R^{r_{m-1}}, where |X_i| = r_i,
        i.e. each component $X_i$ can take on $r_i$ distinct values.
        '''
        res = np.zeros(self.r)
        pt_support_points = [range(r_i) for r_i in self.r]

        for idx in product(*pt_support_points):
            idx_ij = np.array(idx)[self.I + self.J]

            res[idx] = self.pt[tuple(idx_ij)]

        return res
    
