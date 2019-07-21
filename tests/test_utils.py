import numpy as np
from itertools import product

from fake_data_for_learning import utils as ut

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


def test_possible_default_value():
    assert ut.possible_default_value(1)
    assert ~ut.possible_default_value(1.)
    assert ~ut.possible_default_value(-1)
    assert ~ut.possible_default_value('a')


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

def test_generate_random_cpt():
    cpt = ut.generate_random_cpt(3,2)

    # test that entries are non-negative
    assert np.all(cpt > 0)

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