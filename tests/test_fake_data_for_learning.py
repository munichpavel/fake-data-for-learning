"""Tests for `fake_data_for_learning` package."""

import pytest
import numpy as np
from itertools import product

from fake_data_for_learning import BayesianNodeRV as BNRV

class TestBNRVX1cX0:
    r'''
    Test the bayesian node random variable of X = (X0, X1)
    where
    * X0, X1 are binary
    * X admits the graph X0 -> X1
    * X0 has (marinalized) probability table (0.1, 0.9)^t
    * X1 | X0 has conditional probability table
        (0.2, 0.7)
        (0.8, 0.3)
    '''
    #X0
    pt_X0 = np.array([0.1, 0.9])
    rv0 = BNRV('X0', pt_X0)

    def test_set_values(self):
        '''Test setting rv_discrete outcomes member xk'''
        np.testing.assert_equal(self.rv0.values, np.array([0,1]))


    def test_rvs(self):
        assert set(self.rv0.rvs(size=100)).issubset(set(self.rv0.values))

    # # X1 | X0
    # pt_X1cX0 = np.array([
    #     [0.2, 0.7],
    #     [0.8, 0.3]
    # ])

    # rv1c0 = BNRV('X1', pt_X1cX0, parents=['X0'])

    # def test_rvs_1c0(self):
    #     assert set(self.rv1c0.rvs(size=10)).issubset(set(self.rv1c0._xk))


