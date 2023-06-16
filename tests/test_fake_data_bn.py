import pytest

import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal

from fake_data_for_learning.fake_data_for_learning import (
    BayesianNodeRV, FakeDataBayesianNetwork
)


# Test static methods
def test_name_in_list():
    assert FakeDataBayesianNetwork.name_in_list('bob', None) == 0
    assert FakeDataBayesianNetwork.name_in_list('alice', ['alice', 'bob']) == 1


def test_parent_idx():
    X = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])

    assert FakeDataBayesianNetwork.get_parent_idx(2, X) == [0, 1]
    assert FakeDataBayesianNetwork.get_parent_idx(0, X) == []


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
        parent_names=[rv_binary_X0.name]
    )


def test_parents(rv_binary_X0, rv_binary_child_X1):
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
                parent_names=['geoff']
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
                [0.6, 0.4],  # 20, salaried
                [0.1, 0.9],  # 20, self-employed
                [0.2, 0.8],  # 20, student
                [0.3, 0.7],  # 20, unemployed
            ],
            [
                [0.2, 0.8],  # 40, salaried
                [0.3, 0.7],  # 40, self-employed
                [0.7, 0.3],  # 40, student
                [0.4, 0.6],  # 40, unemployed
            ],
            [
                [0.25, 0.75],  # 60, salaried
                [0.3, 0.7],  # 60, self-employed
                [0.2, 0.8],  # 60, student
                [0.1, 0.9],  # 60, unemployed
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
        == (2, 2)
    )

    # X0 -> X2 <- Y1 with Y1 ternary
    assert(
        non_binary_bayesian_network.get_expected_cpt_dims(
            [0, 1], len(non_binary_bayesian_network.bnrvs[2].values)
        )
        == (2, 3, 2)
    )

    # Thriftiness Bayesian network
    assert (
        thrifty_bayesian_network.get_expected_cpt_dims(
            [0, 1], len(thrifty_bayesian_network.bnrvs[2].values)
        )
        == (3, 4, 2)
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


def test_topological_sort():
    eve_0 = BayesianNodeRV('E0', np.array([0.2, 0.8]))
    eve_1 = BayesianNodeRV('E1', np.array([0.7, 0.3]))
    descendent = BayesianNodeRV(
        'D',
        np.array([
            [
                [0.3, 0.7],
                [0.7, 0.3]
            ],
            [
                [0.7, 0.3],
                [0.7, 0.3]
            ]
        ]),
        parent_names=['E0', 'E1']
    )
    further_descendent = BayesianNodeRV(
        'FD',
        np.array([
            [0.4, 0.6],
            [0.6, 0.4]
        ]),
        parent_names=['D']
    )
    bn = FakeDataBayesianNetwork(eve_0, eve_1, descendent, further_descendent)

    res = list(bn.get_topological_ordering())
    assert (
        res == ['E0', 'E1', 'D', 'FD'] or
        res == ['E1', 'E0', 'D', 'FD']
    )


def test_rvs_boundary_cases():
    pt_always_0 = np.array([1., 0.])
    always_0 = BayesianNodeRV('X0', pt_always_0)
    always_1c0 = BayesianNodeRV(
        'X1',
        np.array([
            [0., 1.],
            [1., 0.],
        ]),
        parent_names=['X0']
    )

    bn_1c0 = FakeDataBayesianNetwork(always_0, always_1c0)
    print(bn_1c0.rvs(size=1))
    pd.testing.assert_frame_equal(
        bn_1c0.rvs(size=1),
        pd.DataFrame.from_records([{'X0': 0, 'X1': 1}])
    )

    always_0c0 = BayesianNodeRV(
        'X1',
        np.array([
            [1., 0.],
            [0., 1.],
        ]),
        parent_names=['X0']
    )
    bn_0c0 = FakeDataBayesianNetwork(always_0, always_0c0)
    pd.testing.assert_frame_equal(
        bn_0c0.rvs(size=1),
        pd.DataFrame.from_records([{'X0': 0, 'X1': 0}])
    )


def test_pmf():
    X0 = BayesianNodeRV('X0', np.array([0.1, 0.9]))

    X1cX0 = BayesianNodeRV(
        'X1',
        np.array([
            [0.2, 0.8],
            [0.7, 0.3]
        ]),
        parent_names=[X0.name]
    )
    bn = FakeDataBayesianNetwork(X0, X1cX0)

    assert_almost_equal(
        bn.pmf(pd.Series([0, 0])),
        X0.cpt[0] * X1cX0.cpt[0, 0]
    )
    assert_almost_equal(
        bn.pmf(pd.Series([1, 0])),
        X0.cpt[1] * X1cX0.cpt[1, 0]
    )
    assert_almost_equal(
        bn.pmf(pd.Series([0, 1])),
        X0.cpt[0] * X1cX0.cpt[0, 1]
    )
    assert_almost_equal(
        bn.pmf(pd.Series([1, 1])),
        X0.cpt[1] * X1cX0.cpt[1, 1]
    )


def test_pmf_non_default_values():
    level = BayesianNodeRV('Level', np.array([0.1, 0.9]), values=['high', 'low'])

    outcome = BayesianNodeRV(
        'Outcome',
        np.array([
            [0.2, 0.5, 0.3],
            [0.3, 0.4, 0.3],
        ]),
        values=['bad', 'good', 'meltdown'],
        parent_names=['Level']
    )
    bn = FakeDataBayesianNetwork(level, outcome)

    assert pytest.approx(bn.pmf(pd.Series(['high', 'bad']))) == \
        pytest.approx(level.cpt[0] * outcome.cpt[0, 0])
    assert pytest.approx(bn.pmf(pd.Series(['low', 'meltdown']))) == \
        pytest.approx(level.cpt[1] * outcome.cpt[1, 2])


def test_rvs_counts_vs_pmf():
    # X0 -> X1 binary
    pt_X0 = np.array([0.5, 0.5])
    X0 = BayesianNodeRV('X0', pt_X0)
    pt_X1cX0 = np.array([
            [0.25, 0.75],
            [0.75, 0.25],
    ])
    X1 = BayesianNodeRV('X1', pt_X1cX0, parent_names=['X0'])

    bn = FakeDataBayesianNetwork(X0, X1)
    samples = bn.rvs(size=5000)
    sample_ratios = samples.groupby(['X0', 'X1']).size() / samples.shape[0]

    expected_index = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        names=['X0', 'X1'])
    expected_ratios = pd.Series(
        [0.125, 0.375, 0.375, 0.125],
        index=expected_index
    )
    pd.testing.assert_series_equal(
        sample_ratios, expected_ratios, check_exact=False, rtol=0.1
    )
