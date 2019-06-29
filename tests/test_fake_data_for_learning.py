"""Tests for `fake_data_for_learning` package."""

import pytest
import numpy as np
from itertools import product

from sklearn.preprocessing import LabelEncoder

from fake_data_for_learning import BayesianNodeRV as BNRV
from fake_data_for_learning import FakeDataBayesianNetwork as FDBN
import fake_data_for_learning.utils as ut


class TestBNRVX0:
    #X0
    pt_X0 = np.array([0.1, 0.9])
    rv0 = BNRV('X0', pt_X0)

    def test_equality(self):
        other_rv0 = BNRV('X0', self.pt_X0)
        assert self.rv0 == other_rv0

    def test_default_set_values(self):
        '''Test setting outcome values if None as argument'''
        np.testing.assert_equal(self.rv0.values, np.array([0,1]))

    def test_default_rv_values(self):
        assert isinstance(self.rv0.rvs(seed=42), np.int64)
        assert len(self.rv0.rvs(size=100)) == 100

        # Verify sample result for fixed seed
        expected_draw_X0 = 1
        assert self.rv0.rvs(seed=42) == expected_draw_X0

    def test_rv_argument_handling(self):
        expected_draw_X0 = 1
        # Test handling of extraneous connditioned values
        assert self.rv0.rvs(parent_values={'X1': 1}, seed=42) == expected_draw_X0
    
    def test_get_pt(self):
        np.testing.assert_equal(self.rv0.get_pt(), self.pt_X0)

    # Test non-default random variable values
    rv0_char = BNRV('X0', pt_X0, values=['up', 'down'])

    def test_set_values(self):
        assert len(self.rv0_char.values) == 2
        assert set(self.rv0_char.values) == set(['up', 'down'])
        # Ordering of internal representation must match initial value ordering
        np.testing.assert_equal(
            self.rv0_char._values,
            np.array([0, 1])
        )

    def test_wonky_nondef_values(self):
        with pytest.raises(ValueError):
            BNRV('X0', self.pt_X0, values=['up_up_away', 'down'])

    def test_rvs(self):
        assert isinstance(self.rv0_char.rvs(seed=42), str)



class TestBNRVX1cX0:
    r'''
    Test the bayesian node random variable of X = (X0, X1)
    where
    * X0, X1 are binary
    * X admits the graph X0 -> X1
    * X1 | X0 has conditional probability table
        np.array([
            [0.2, 0.8],
            [0.7, 0.3]
        ])
    '''
    # X1 | X0
    pt_X1cX0 = np.array([
        [0.2, 0.8],
        [0.7, 0.3]
    ])

    rv1c0 = BNRV('X1', pt_X1cX0, parent_names=['X0'])

    def test_set_values_1c0(self):
        np.testing.assert_equal(self.rv1c0.values, np.array([0,1]))

    def test_rvs_1c0(self):
        draw = self.rv1c0.rvs(parent_values={'X0': 1}, seed=42)
        assert isinstance(draw, np.int64)
        # Verify result for fixed seed
        expected_X1cX0_draw = 0
        assert draw == expected_X1cX0_draw

    def test_rvs_argument_handling_1c0(self):
        # Verify result for fixed seed
        expected_X1cX0_draw = 0
        # Test handling of extraneous conditioned values
        draw = self.rv1c0.rvs(parent_values={'X0': 1, 'X2': 42}, seed=42)
        assert draw == expected_X1cX0_draw

    def test_get_pt(self):
        res = self.rv1c0.get_pt(parent_values={'X0': 1})
        np.testing.assert_equal(res, self.pt_X1cX0[1, :])

    # With non-devault values for X0
    pt_X0 = np.array([0.1, 0.9])
    rv0_nondef = BNRV('X0', pt_X0, values=['up', 'down'])

    def test_get_pt_nondef(self):
        # parent values has external value and label encoder
        res = self.rv1c0.get_pt(
            parent_values={
                'X0': {
                    'value': 'down',
                    'le': self.rv0_nondef.le
                }
            }
        )
        np.testing.assert_equal(res, self.pt_X1cX0[1, :])
        

class TestBNRVX2cX0X1:
    r'''
    Test the bayesian node random variable X = (X0, X1, X2)
    where
    * X0, X1, X2 are binary
    * X admits the graph X0 -> X2 <- X1
    * X2 | X0, X1 has conditional probability table
        np.array([
            [
                [0., 1.],
                [0.5, 0.5]
            ],
            [
                [0.9, 0.1],
                [0.3, 0.7]
            ]
        ])
    '''
    # X2 | X0, X1
    pt_X2cX0X1 = np.array([
        [
            [0., 1.],
            [0.5, 0.5]
        ],
        [
            [0.9, 0.1],
            [0.3, 0.7]
        ]
    ])


    rv2c01 = BNRV('X2', pt_X2cX0X1, parent_names=['X0', 'X1'])

    def test_rvs_2c00(self):
        draw = self.rv2c01.rvs(parent_values={'X0': 0, 'X1': 0})
        assert isinstance(draw, np.int64)

    def test_get_pt(self):
        res = self.rv2c01.get_pt(parent_values={'X0': 0, 'X1': 0})
        np.testing.assert_equal(res, self.pt_X2cX0X1[0, 0, :])



class TestFakeDataBayesianNetwork:

    ###############################
    #  Bayesian network X0 -> X1
    # X0, X1 binary
    ###############################
    
    #X0
    pt_X0 = np.array([0.1, 0.9])
    rv0 = BNRV('X0', pt_X0)

    # X1 | X0
    pt_X1cX0 = np.array([
        [0.2, 0.8],
        [0.7, 0.3]
    ])

    rv1c0 = BNRV('X1', pt_X1cX0, parent_names=['X0'])

    def test_missing_names(self):
        with pytest.raises(ValueError):
            FDBN(self.rv1c0)

    bn = FDBN(rv0, rv1c0)

    def test_node_names(self):
        assert len(self.bn.node_names) == 2
        assert set(self.bn.node_names) == set(['X0', 'X1'])
    
    def test_adjacency(self):
        expected_adjacency = np.array([
            [0, 1],
            [0, 0]
        ])
        np.testing.assert_equal(
            self.bn.adjacency_matrix,
            expected_adjacency
        )

    # # With non-default valued outocomes
    # rv0_nondef = BNRV('X0', pt_X0, values=['a', 'b'])
    # rv1c0_nondef = BNRV('X1', pt_X1cX0, parent_names=['X0'], values=['male', 'female'])

    # bn_nondef = FDBN(rv0_nondef, rv1c0_nondef)
    # def test_rvs_type_nondef_values(self):
    #     sample = self.bn_nondef.rvs(seed=42)
    #     assert sample.dtype.type is np.unicode_


    ###############################
    #  Bayesian network X0 -> X2 <- X1
    # X0, X2 binary, X1 ternary
    ###############################
    # X1
    pt_X1 = np.array([0.7, 0.2, 0.1])
    rv1 = BNRV('X1', pt_X1)

    # X2 | X0, X1
    pt_X2cX0X1 = np.array([
        [
            [0., 1.0],
            [0.2, 0.8],
            [0.1, 0.9]
        ],
        [
            [0.5, 0.5],
            [0.3, 0.7],
            [0.9, 0.1]
        ],

    ])

    rv2c01 = BNRV('X2', pt_X2cX0X1, parent_names=['X0', 'X1'])

    bn2c01 = FDBN(rv0, rv1, rv2c01)
    
    def test_2c01_adjacency(self):
        expected_adjacency = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        np.testing.assert_equal(
            self.bn2c01.adjacency_matrix,
            expected_adjacency
        )

    def test_eve_node_names(self):
        assert sorted(self.bn2c01._eve_node_names) == ['X0', 'X1']

    ###############################
    #  Bayesian network 
    # X0 -> X2 <- X1
    #        |
    #         --> X3
                
    # X0, X1, X2, X3 binary
    ###############################
    # X3 | X2
    cpt_X3cX2 = np.array([
        [0.25, 0.75],
        [0., 1.]
    ])
    rv_3c2 = BNRV('X3', cpt=cpt_X3cX2, parent_names=['X2'])
    bn3c2c01 = FDBN(rv0, rv1, rv2c01, rv_3c2)

    def test_3c2c01_adjacency(self):
        expected_adjacency = np.array([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])

        np.testing.assert_equal(
            self.bn3c2c01.adjacency_matrix,
            expected_adjacency
        )

    def test_3c2c01_eve_node_names(self):
        assert sorted(self.bn2c01._eve_node_names) == ['X0', 'X1']

    def test_3c2c01_rvs(self):
        expected_was_sampled = np.array(4 * [True])
        sample = self.bn3c2c01.rvs(seed=42)
        np.testing.assert_equal(~np.isnan(sample), expected_was_sampled)


    ###############################
    #  Bayesian network X0 -> X1
    # X0 tertiary, X1 binary
    ###############################
    
    #X0
    pt_X0 = np.array([0.1, 0.7, 0.2])
    rv0 = BNRV('X0', pt_X0)

    

    def test_bn_matrix_dimensions(self):
        # X1 | X0
        pt_X1cX0_wrong_dims = np.array([
            [0.2, 0.8],
            [0.7, 0.3],
        ])
        rv1c0_wrong_dims = BNRV('X1', pt_X1cX0_wrong_dims, parent_names=['X0'])
        with pytest.raises(ValueError):
            FDBN(self.rv0, rv1c0_wrong_dims)


###############
# Test utils
###############
def test_name_in_list():
        assert ut.name_in_list('bob', None) == 0
        assert ut.name_in_list('alice', ['alice', 'bob']) == 1


def test_zero_column_idx():
    X = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    expected_idx = np.array([0, 2])
    np.testing.assert_equal(ut.zero_column_idx(X), expected_idx)


def test_parent_idx():
    X = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])    

    assert ut.get_parent_idx(2, X) == [0,1]
    assert ut.get_parent_idx(0, X) == []

def test_get_pure_descendent_idx():
    # Test with graph
    # X0        X1--
    # |         |   |
    #  -> X2 <--    |
    #      |        |
    # X3 <- -> X4 <-
    X = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    # Pure descendent of X0, X1 is X2
    np.testing.assert_equal(
        ut.get_pure_descendent_idx(np.array([0,1]), X),
        np.array([2])
    )
    # Pure descendent of X2 is X3
    np.testing.assert_equal(
        ut.get_pure_descendent_idx(np.array([2]), X),
        np.array([3])
    )

    # No pure descendents of X1
    np.testing.assert_equal(
        ut.get_pure_descendent_idx(np.array([1]), X),
        np.array([])
    )
