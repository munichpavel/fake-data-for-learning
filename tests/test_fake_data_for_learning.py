"""Tests for `fake_data_for_learning` package."""

import pytest
import numpy as np
from itertools import product

from fake_data_for_learning import BayesianNodeRV as BNRV
from fake_data_for_learning import FakeDataBayesianNetwork as FDBN


class TestBNRVX0:
    #X0
    pt_X0 = np.array([0.1, 0.9])
    rv0 = BNRV('X0', pt_X0)

    def test_default_set_values(self):
        '''Test setting outcome values if None as argument'''
        np.testing.assert_equal(self.rv0.values, np.array([0,1]))


    def test_default_rvs(self):
        assert isinstance(self.rv0.rvs(seed=42), np.int64)
        assert len(self.rv0.rvs(size=100)) == 100

    
    def test_get_pt(self):
        np.testing.assert_equal(self.rv0.get_pt(), self.pt_X0)


    rv0_char = BNRV('X0', pt_X0, values=['down', 'up'])

    def test_set_values(self):
        assert len(self.rv0_char.values) == 2
        assert set(self.rv0_char.values) == set(['down', 'up'])


    def test_rvs(self):
        assert isinstance(self.rv0_char.rvs(seed=42), str)



class TestBNRVX1cX0:
    r'''
    Test the bayesian node random variable of X = (X0, X1)
    where
    * X0, X1 are binary
    * X admits the graph X0 -> X1
    * X1 | X0 has conditional probability table
        (0.2, 0.7)
        (0.8, 0.3)
    '''
    
 
    # X1 | X0
    pt_X1cX0 = np.array([
        [0.2, 0.7],
        [0.8, 0.3]
    ])

    rv1c0 = BNRV('X1', pt_X1cX0, parents=['X0'])

    def test_set_values_1c0(self):
        np.testing.assert_equal(self.rv1c0.values, np.array([0,1]))


    def test_rvs_1c0(self):
        draw = self.rv1c0.rvs(parent_values={'X0': 1})
        assert isinstance(draw, np.int64)


    def test_get_pt(self):
        res = self.rv1c0.get_pt(parent_values={'X0': 1})
        np.testing.assert_equal(res, self.pt_X1cX0[:, 1])


class TestFakeDataBayesianNetwork:

    #X0
    pt_X0 = np.array([0.1, 0.9])
    rv0 = BNRV('X0', pt_X0)

    # X1 | X0
    pt_X1cX0 = np.array([
        [0.2, 0.7],
        [0.8, 0.3]
    ])

    rv1c0 = BNRV('X1', pt_X1cX0, parents=['X0'])


    def test_missing_names(self):
        with pytest.raises(ValueError):
            FDBN(self.rv1c0)


    bn = FDBN(rv0, rv1c0)

    def test_node_names(self):
        assert len(self.bn.node_names) == 2
        assert set(self.bn.node_names) == set(['X0', 'X1'])


    def test_name_in_list(self):
        assert self.bn._name_in_list('bob', None) == 0
        assert self.bn._name_in_list('alice', ['alice', 'bob']) == 1
    
 
    def test_adjacency(self):
        expected_adjacency = np.array([
            [0, 1],
            [0, 0]
        ])
        np.testing.assert_equal(
            self.bn.adjacency_matrix,
            expected_adjacency
        )
