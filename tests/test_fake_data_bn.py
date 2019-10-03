import pytest

import numpy as np
from itertools import product

from fake_data_for_learning import BayesianNodeRV
from fake_data_for_learning import FakeDataBayesianNetwork
from fake_data_for_learning import SampleValue

from fake_data_for_learning import utils as ut


# Test static methods
def test_name_in_list():
        assert FakeDataBayesianNetwork.name_in_list('bob', None) == 0
        assert FakeDataBayesianNetwork.name_in_list('alice', ['alice', 'bob']) == 1


def test_zero_column_idx():
    X = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    expected_idx = np.array([0, 2])
    np.testing.assert_equal(FakeDataBayesianNetwork.zero_column_idx(X), expected_idx)


def test_parent_idx():
    X = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])    

    assert FakeDataBayesianNetwork.get_parent_idx(2, X) == [0,1]
    assert FakeDataBayesianNetwork.get_parent_idx(0, X) == []


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
        FakeDataBayesianNetwork.get_pure_descendent_idx(np.array([0,1]), X),
        np.array([2])
    )
    # Pure descendent of X2 is X3
    np.testing.assert_equal(
        FakeDataBayesianNetwork.get_pure_descendent_idx(np.array([2]), X),
        np.array([3])
    )

    # No pure descendents of X1
    np.testing.assert_equal(
        FakeDataBayesianNetwork.get_pure_descendent_idx(np.array([1]), X),
        np.array([])
    )

def test_non_zero_column_idx():
    X = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1]
    ])
    np.testing.assert_equal(
        FakeDataBayesianNetwork.non_zero_column_idx(X),
        np.array([2,3,4])
    )


def test_generate_random_cpt():
    cpt = ut.generate_random_cpt(3,2)

    # test that entries are non-negative
    assert np.all(cpt >= 0)

def test_make_cpt():
    cpt = ut.make_cpt(np.random.rand(3,2,4))

    # test that final dimensions sum to 1
    ranges = [range(s) for s in cpt.shape[:-1]]
    for r in product(*ranges):
        np.testing.assert_almost_equal(sum(cpt[r]), 1)

    cpt_from_negative = ut.make_cpt(-np.ones((4,2,3)))

    # test that final dimensions sum to 1
    ranges = [range(s) for s in cpt_from_negative.shape[:-1]]
    for r in product(*ranges):
        np.testing.assert_almost_equal(sum(cpt_from_negative[r]), 1)


@pytest.fixture
def rv_binary_X0():
    return BayesianNodeRV('X0', np.array([0.1, 0.9]))

@pytest.fixture
def rv_binary_child_X1(rv_binary_X0):
    return BayesianNodeRV(
        'X1', 
        np.array([
            [0.2, 0.8],
            [0.7, 0.3]
        ]),
        parent_names = [rv_binary_X0.name]
    )


def test_parents(
    rv_binary_X0, rv_binary_child_X1  
):
    
    # Test for missing parent variable
    with pytest.raises(ValueError):
        FakeDataBayesianNetwork(rv_binary_child_X1)

    # Test for wrong parent name
    with pytest.raises(ValueError):
        FakeDataBayesianNetwork(
            rv_binary_child_X1,
            BayesianNodeRV(
                'X1',
                np.array([
                    [0.2, 0.8],
                    [0.7, 0.3]
                ]),
                parent_names = ['geoff']
            )
        )


@pytest.fixture
def non_binary_bayesian_network(rv_binary_X0):
    # X0 -> X2 <- Y1
    return FakeDataBayesianNetwork(
        rv_binary_X0,
        BayesianNodeRV('Y1', np.array([0.1, 0.7, 0.2])),
        BayesianNodeRV(
            'X2',
            np.array([
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

            ]),
            parent_names=['X0', 'Y1']
        )
    )


@pytest.fixture
def thrifty_bayesian_network():
    r'''
        age     ->      profession
        |                   /
        -> thriftiness <---
    '''
    age = BayesianNodeRV('age', np.array([0.2, 0.5, 0.3]), values=('20', '40', '60'))
    profession = BayesianNodeRV(
        'profession', 
        np.array([
            [0.3, 0.4, 0.2, 0.1],
            [0.05, 0.15, 0.3, 0.5],
            [0.15, 0.05, 0.2, 0.6]
        ]),
        values=('salaried', 'self-employed', 'student', 'unemployed'),
        parent_names=['age'])

    thriftiness = BayesianNodeRV(
        'thriftiness',
        np.array([
            [
                [0.6, 0.4], #20, salaried
                [0.1, 0.9], #20, self-employed
                [0.2, 0.8], #20, student
                [0.3, 0.7], #20, unemployed
            ],
            [
                [0.2, 0.8], # 40 salaried
                [0.3, 0.7], # 40, self-employed
                [0.7, 0.3], #40, student
                [0.4, 0.6], #40, unemployed                
            ],
            [
                [0.25, 0.75], #60, salaried
                [0.3, 0.7], #60, self-employed
                [0.2, 0.8], #60, student
                [0.1, 0.9], #60, unemployed
            ],
        ]),
        parent_names=['age', 'profession']
    )

    return FakeDataBayesianNetwork(age, profession, thriftiness)


def test_expected_cpt_dims(
    rv_binary_X0, rv_binary_child_X1,
    non_binary_bayesian_network,
    thrifty_bayesian_network
):
    bn = FakeDataBayesianNetwork(rv_binary_X0, rv_binary_child_X1)
    assert (
        bn.get_expected_cpt_dims([0], len(rv_binary_X0.values))
        == [2,2]
    )

    # X0 -> X2 <- Y1 with Y1 ternary
    assert(
        non_binary_bayesian_network,
        [2,3,2]
    )

    # Thriftiness Bayesian network
    assert (
        thrifty_bayesian_network.get_expected_cpt_dims(
            [0,1], len(thrifty_bayesian_network.bnrvs[2].values)
        )
        ==  [3,4,2]
    )



def test_adjacency_matrix(
    rv_binary_X0, rv_binary_child_X1,
    non_binary_bayesian_network,
    thrifty_bayesian_network
):
    bn = FakeDataBayesianNetwork(rv_binary_X0, rv_binary_child_X1)
    expected_adj = np.array(
        [[0, 1], [0, 0]]
    )

    np.testing.assert_equal(
        bn.adjacency_matrix,
        expected_adj
    )

    # X0 -> X2 <- Y1
    expected_nonbinary_adj = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    np.testing.assert_equal(
        non_binary_bayesian_network.adjacency_matrix,
        expected_nonbinary_adj
    )
   
    expected_thrifty_adj = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    np.testing.assert_equal(
        thrifty_bayesian_network.adjacency_matrix,
        expected_thrifty_adj
    )

def test_ancestral_sampling(
    non_binary_bayesian_network,
    thrifty_bayesian_network
):
    # Test eve names, i.e. nodes with no parents
    assert (
        non_binary_bayesian_network.eve_node_names
        == ['X0', 'Y1']
    )

    assert(
        thrifty_bayesian_network.eve_node_names
        == ['age']
    )

    #Test all nodes sampled
    assert not non_binary_bayesian_network.all_nodes_sampled(
        {
            'X0': SampleValue(0),
            'Y0': SampleValue(2)
        }
    )
    assert thrifty_bayesian_network.all_nodes_sampled(
        {
            'age': SampleValue(
                '20', thrifty_bayesian_network.bnrvs[0].label_encoder
            ),
            'profession': SampleValue(
                'unemployed', thrifty_bayesian_network.bnrvs[1].label_encoder
            ),
            'thriftiness': SampleValue(1)
        }
    )



##############################
# # Test FakeDataBayesianNetwork
# ##############################

# class TestFakeDataBayesianNetwork:

#     ###############################
#     #  Bayesian network X0 -> X1
#     # X0, X1 binary
#     ###############################
    
#     #X0
#     pt_X0 = np.array([0.1, 0.9])
#     rv0 = BNRV('X0', pt_X0)

#     # X1 | X0
#     pt_X1cX0 = np.array([
#         [0.2, 0.8],
#         [0.7, 0.3]
#     ])

#     rv1c0 = BNRV('X1', pt_X1cX0, parent_names=['X0'])

#     def test_missing_names(self):
#         with pytest.raises(ValueError):
#             FDBN(self.rv1c0)

#     bn = FDBN(rv0, rv1c0)

#     def test_node_names(self):
#         assert len(self.bn.node_names) == 2
#         assert set(self.bn.node_names) == set(['X0', 'X1'])
    
#     def test_adjacency(self):
#         expected_adjacency = np.array([
#             [0, 1],
#             [0, 0]
#         ])
#         np.testing.assert_equal(
#             self.bn.adjacency_matrix,
#             expected_adjacency
#         )

#     # With non-default valued outcomes
#     rv0_nondef = BNRV('X0', pt_X0, values=['a', 'b'])
#     rv1c0_nondef = BNRV('X1', pt_X1cX0, parent_names=['X0'], values=['male', 'female'])

#     bn_nondef = FDBN(rv0_nondef, rv1c0_nondef)

#     def test_rvs_type_nondef_values(self):
#         sample = self.bn_nondef.rvs(seed=42)
#         expected_sample = pd.DataFrame({'X0': 'b', 'X1': 'male'}, index=range(1), columns=('X0', 'X1'))
#         pd.testing.assert_frame_equal(sample, expected_sample)

#         samples = self.bn_nondef.rvs(size=2, seed=42)
#         expected_samples = pd.DataFrame(
#             {'X0': ['b', 'b'], 'X1': ['male', 'male']}, index=range(2), columns=('X0', 'X1'))
#         pd.testing.assert_frame_equal(samples, expected_samples)

#     ###############################
#     #  Bayesian network X0 -> X2 <- X1
#     # X0, X2 binary, X1 ternary
#     ###############################
#     # X1
#     pt_X1 = np.array([0.7, 0.2, 0.1])
#     rv1 = BNRV('X1', pt_X1)

#     # X2 | X0, X1
#     pt_X2cX0X1 = np.array([
#         [
#             [0., 1.0],
#             [0.2, 0.8],
#             [0.1, 0.9]
#         ],
#         [
#             [0.5, 0.5],
#             [0.3, 0.7],
#             [0.9, 0.1]
#         ],

#     ])

#     rv2c01 = BNRV('X2', pt_X2cX0X1, parent_names=['X0', 'X1'])

#     bn2c01 = FDBN(rv0, rv1, rv2c01)
    
#     def test_2c01_adjacency(self):
#         expected_adjacency = np.array([
#             [0, 0, 1],
#             [0, 0, 1],
#             [0, 0, 0]
#         ])
#         np.testing.assert_equal(
#             self.bn2c01.adjacency_matrix,
#             expected_adjacency
#         )

#     def test_eve_node_names(self):
#         assert sorted(self.bn2c01._eve_node_names) == ['X0', 'X1']

#     ###############################
#     #  Bayesian network 
#     # X0 -> X2 <- X1
#     #        |
#     #         --> X3
                
#     # X0, X1, X2, X3 binary
#     ###############################
#     # X3 | X2
#     cpt_X3cX2 = np.array([
#         [0.25, 0.75],
#         [0., 1.]
#     ])
#     rv_3c2 = BNRV('X3', cpt=cpt_X3cX2, parent_names=['X2'])
#     bn3c2c01 = FDBN(rv0, rv1, rv2c01, rv_3c2)

#     def test_3c2c01_adjacency(self):
#         expected_adjacency = np.array([
#             [0, 0, 1, 0],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1],
#             [0, 0, 0, 0]
#         ])

#         np.testing.assert_equal(
#             self.bn3c2c01.adjacency_matrix,
#             expected_adjacency
#         )

#     def test_3c2c01_eve_node_names(self):
#         assert sorted(self.bn2c01._eve_node_names) == ['X0', 'X1']

#     def test_3c2c01_rvs(self):
#         expected_was_sampled = np.array([4 * [True]])
#         sample = self.bn3c2c01.rvs(seed=42)
#         np.testing.assert_equal(~np.isnan(sample), expected_was_sampled)


#     ###############################
#     #  Bayesian network X0 -> X1
#     # X0 tertiary, X1 binary
#     ###############################
    
#     #X0
#     pt_X0 = np.array([0.1, 0.7, 0.2])
#     rv0 = BNRV('X0', pt_X0)

    

#     def test_bn_matrix_dimensions(self):
#         # X1 | X0
#         pt_X1cX0_wrong_dims = np.array([
#             [0.2, 0.8],
#             [0.7, 0.3],
#         ])
#         rv1c0_wrong_dims = BNRV('X1', pt_X1cX0_wrong_dims, parent_names=['X0'])
#         with pytest.raises(ValueError):
#             FDBN(self.rv0, rv1c0_wrong_dims)

   
#     ###########################################################################
#     # Bayesian network age ->               thriftiness
#     #                     \> employment />
#     # age takes 3 values, employment takes 4 values, and thriftiness if binary
#     ###########################################################################
#     age = BNRV('age', np.array([0.2, 0.5, 0.3]), values=('20', '40', '60'))

#     profession = BNRV(
#         'profession', 
#         np.array([
#             [0.3, 0.4, 0.2, 0.1],
#             [0.05, 0.15, 0.3, 0.5],
#             [0.15, 0.05, 0.2, 0.6]
#         ]),
#         values=('unemployed', 'student', 'self-employed', 'salaried'),
#         parent_names=['age'])

#     thriftiness = BNRV(
#         'thriftiness',
#         np.array([
#             [
#                 [0.3, 0.7], #20, unemployed
#                 [0.2, 0.8], #20, student
#                 [0.1, 0.9], #20, self-employed
#                 [0.6, 0.4], #20, salaried
#             ],
#             [
#                 [0.4, 0.6], #40, unemployed
#                 [0.7, 0.3], #40, student
#                 [0.3, 0.7], # 40, self-employed
#                 [0.2, 0.8], # 40 salaried
#             ],
#             [
#                 [0.1, 0.9], #60, unemployed
#                 [0.2, 0.8], #60, student
#                 [0.3, 0.7], #60, self-employed
#                 [0.25, 0.75], #60, salaried
#             ],
#         ]),
#         parent_names=['age', 'profession']
#     )


#     thrifty_bn = FDBN(age, profession, thriftiness)
#     def test_get_node(self):
#         assert self.thrifty_bn.get_node('age') == self.age
        
#         with pytest.raises(ValueError):
#             self.thrifty_bn.get_node('pompitousness')

#     def test_expected_3d_cpt_dimension(self):
#        assert (
#             self.thrifty_bn.get_expected_cpt_dims([0,1], len(self.thrifty_bn.bnrvs[2].values))
#             ==  [3,4,2]
#         )

#     samples_partial = {'age': SampleValue('20', LabelEncoder())}
#     samples_all = {
#             'age': SampleValue('20', LabelEncoder()),
#             'profession': SampleValue('student', LabelEncoder()),
#             'thriftiness': SampleValue(0)
#         }
#     def test_all_nodes_sampled(self):
#         assert ~self.thrifty_bn.all_nodes_sampled(self.samples_partial)
#         assert self.thrifty_bn.all_nodes_sampled(self.samples_all)

#     def test_all_parents_sampled(self):
#         assert self.thrifty_bn.all_parents_sampled('age', {})
#         assert ~self.thrifty_bn.all_parents_sampled('thriftiness', self.samples_partial)
#         assert self.thrifty_bn.all_parents_sampled('thriftiness', self.samples_all)
        
#     def test_get_unsampled_nodes(self):
#         assert (
#             set(self.thrifty_bn.get_unsampled_nodes(self.samples_partial))
#             == {'profession', 'thriftiness'}
#         )

#     def test_thriftiness_rvs(self):
#         sample = self.thrifty_bn.rvs(seed=42)
#         expected_sample = pd.DataFrame(
#             {
#                 'age': '40',
#                 'profession': 'self-employed',
#                 'thriftiness': 1
#             }, 
#             index=range(1), columns=('age', 'profession', 'thriftiness'))
#         pd.testing.assert_frame_equal(sample, expected_sample)