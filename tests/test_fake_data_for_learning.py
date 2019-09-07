"""Tests for `fake_data_for_learning` package."""

import pytest
import numpy as np
import pandas as pd
from itertools import product

from sklearn.preprocessing import LabelEncoder

from fake_data_for_learning import BayesianNodeRV as BNRV
from fake_data_for_learning import FakeDataBayesianNetwork as FDBN
from fake_data_for_learning import SampleValue
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
        assert self.rv0.rvs(parent_values={'X1': SampleValue(1)}, seed=42) == expected_draw_X0
    
    def test_get_pt(self):
        np.testing.assert_equal(self.rv0.get_pt(), self.pt_X0)

    # Test non-default random variable values
    rv0_char = BNRV('X0', pt_X0, values=['up', 'down'])

    def test_set_values(self):
        assert len(self.rv0_char.values) == 2
        np.testing.assert_equal(
            self.rv0_char.values,
            np.array(['up', 'down'], dtype=object)
        )

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
        draw = self.rv1c0.rvs(parent_values={'X0': SampleValue(1)}, seed=42)
        assert ut.possible_default_value(draw)
        # Verify result for fixed seed
        expected_X1cX0_draw = 0
        assert draw == expected_X1cX0_draw

    def test_rvs_argument_handling_1c0(self):
        # Verify result for fixed seed
        expected_X1cX0_draw = 0
        # Test handling of extraneous conditioned values
        draw = self.rv1c0.rvs(parent_values={'X0': SampleValue(1), 'X2': SampleValue(42)}, seed=42)
        assert draw == expected_X1cX0_draw

    def test_get_pt(self):
        res = self.rv1c0.get_pt(parent_values={'X0': SampleValue(1)})
        np.testing.assert_equal(res, self.pt_X1cX0[1, :])

    # With non-devault values for X0
    pt_X0 = np.array([0.1, 0.9])
    rv0_nondef = BNRV('X0', pt_X0, values=['up', 'down'])

    def test_get_pt_nondef(self):
        res = self.rv1c0.get_pt(
            parent_values={'X0': SampleValue('down', label_encoder=self.rv0_nondef.label_encoder)}
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
            [0.5, 0.5],
        ],
        [
            [0.9, 0.1],
            [0.3, 0.7]
        ]
    ])


    rv2c01 = BNRV('X2', pt_X2cX0X1, parent_names=['X0', 'X1'])

    def test_rvs_2c00(self):
        draw = self.rv2c01.rvs(parent_values={'X0': SampleValue(0), 'X1': SampleValue(0)})
        assert ut.possible_default_value(draw)

    def test_get_pt(self):
        res = self.rv2c01.get_pt(parent_values={'X0': SampleValue(0), 'X1': SampleValue(0)})
        np.testing.assert_equal(res, self.pt_X2cX0X1[0, 0, :])

class TestProfession:
    profession = BNRV(
        'profession', 
        np.array([
            [0.3, 0.4, 0.2, 0.1],
            [0.05, 0.15, 0.3, 0.5],
            [0.15, 0.05, 0.2, 0.6]
        ]),
        values=('unemployed', 'student', 'self-employed', 'salaried'),
        parent_names=['age'])

    
###################
# Test SampleValues
###################

class TestSampleValue:
    with pytest.raises(ValueError):
        SampleValue('a')


##############################
# Test FakeDataBayesianNetwork
##############################

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

    # With non-default valued outcomes
    rv0_nondef = BNRV('X0', pt_X0, values=['a', 'b'])
    rv1c0_nondef = BNRV('X1', pt_X1cX0, parent_names=['X0'], values=['male', 'female'])

    bn_nondef = FDBN(rv0_nondef, rv1c0_nondef)

    def test_rvs_type_nondef_values(self):
        sample = self.bn_nondef.rvs(seed=42)
        expected_sample = pd.DataFrame({'X0': 'b', 'X1': 'male'}, index=range(1), columns=('X0', 'X1'))
        pd.testing.assert_frame_equal(sample, expected_sample)

        samples = self.bn_nondef.rvs(size=2, seed=42)
        expected_samples = pd.DataFrame(
            {'X0': ['b', 'b'], 'X1': ['male', 'male']}, index=range(2), columns=('X0', 'X1'))
        pd.testing.assert_frame_equal(samples, expected_samples)

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
        expected_was_sampled = np.array([4 * [True]])
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

   
    ###########################################################################
    # Bayesian network age ->               thriftiness
    #                     \> employment />
    # age takes 3 values, employment takes 4 values, and thriftiness if binary
    ###########################################################################
    age = BNRV('age', np.array([0.2, 0.5, 0.3]), values=('20', '40', '60'))

    profession = BNRV(
        'profession', 
        np.array([
            [0.3, 0.4, 0.2, 0.1],
            [0.05, 0.15, 0.3, 0.5],
            [0.15, 0.05, 0.2, 0.6]
        ]),
        values=('unemployed', 'student', 'self-employed', 'salaried'),
        parent_names=['age'])

    thriftiness = BNRV(
        'thriftiness',
        np.array([
            [
                [0.3, 0.7], #20, unemployed
                [0.2, 0.8], #20, student
                [0.1, 0.9], #20, self-employed
                [0.6, 0.4], #20, salaried
            ],
            [
                [0.4, 0.6], #40, unemployed
                [0.7, 0.3], #40, student
                [0.3, 0.7], # 40, self-employed
                [0.2, 0.8], # 40 salaried
            ],
            [
                [0.1, 0.9], #60, unemployed
                [0.2, 0.8], #60, student
                [0.3, 0.7], #60, self-employed
                [0.25, 0.75], #60, salaried
            ],
        ]),
        parent_names=['age', 'profession']
    )


    thrifty_bn = FDBN(age, profession, thriftiness)
    def test_get_node(self):
        assert self.thrifty_bn.get_node('age') == self.age
        
        with pytest.raises(ValueError):
            self.thrifty_bn.get_node('pompitousness')

    def test_expected_3d_cpt_dimension(self):
       assert (
            self.thrifty_bn.get_expected_cpt_dims([0,1], len(self.thrifty_bn._bnrvs[2].values))
            ==  [3,4,2]
        )

    samples_partial = {'age': SampleValue('20', LabelEncoder())}
    samples_all = {
            'age': SampleValue('20', LabelEncoder()),
            'profession': SampleValue('student', LabelEncoder()),
            'thriftiness': SampleValue(0)
        }
    def test_all_nodes_sampled(self):
        assert ~self.thrifty_bn.all_nodes_sampled(self.samples_partial)
        assert self.thrifty_bn.all_nodes_sampled(self.samples_all)

    def test_all_parents_sampled(self):
        assert self.thrifty_bn.all_parents_sampled('age', {})
        assert ~self.thrifty_bn.all_parents_sampled('thriftiness', self.samples_partial)
        assert self.thrifty_bn.all_parents_sampled('thriftiness', self.samples_all)
        
    def test_get_unsampled_nodes(self):
        assert (
            set(self.thrifty_bn.get_unsampled_nodes(self.samples_partial))
            == {'profession', 'thriftiness'}
        )

    def test_thriftiness_rvs(self):
        sample = self.thrifty_bn.rvs(seed=42)
        expected_sample = pd.DataFrame(
            {
                'age': '40',
                'profession': 'self-employed',
                'thriftiness': 1
            }, 
            index=range(1), columns=('age', 'profession', 'thriftiness'))
        pd.testing.assert_frame_equal(sample, expected_sample)
