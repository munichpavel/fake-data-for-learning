"""Tests for `fake_data_for_learning` package."""

import pytest
import numpy as np

from sklearn.preprocessing import LabelEncoder

from fake_data_for_learning.fake_data_for_learning import (
    BayesianNodeRV, SampleValue
)

# (Conditional) probability distributions
@pytest.fixture
def binary_pt():
    return np.array([0.1, 0.9])


@pytest.fixture
def binary_cpt():
    return np.array([
        [0.2, 0.8],
        [0.7, 0.3]
    ])


def test_init(binary_pt, binary_cpt):

    # Successful initialization
    BayesianNodeRV('X0', binary_pt)
    BayesianNodeRV('X1', binary_cpt, parent_names=['X0'])
    BayesianNodeRV(
        'profession',
        np.array([
            [0.3, 0.4, 0.2, 0.1],
            [0.05, 0.15, 0.3, 0.5],
            [0.15, 0.05, 0.2, 0.6]
        ]),
        values=('salaried', 'self-employed',  'student', 'unemployed'),
        parent_names=['tertiary-rv']
    )

    # Failing initialization: Parent names must be list
    with pytest.raises(TypeError):
        BayesianNodeRV('X0', binary_cpt, parent_names='X1')

    # Number of parent names must be compatible with shape of cpt
    with pytest.raises(ValueError):
        BayesianNodeRV('X1', binary_cpt)

    with pytest.raises(ValueError):
        BayesianNodeRV('X2', binary_cpt, parent_names=['X0', 'X1'])


def test_encoding(binary_pt):
    # Default values
    binary_rv = BayesianNodeRV('X0', binary_pt)
    np.testing.assert_equal(
        binary_rv.values,
        np.array([0, 1])
    )

    # Non-default values
    binary_rv_nondef = BayesianNodeRV(
        'X0', binary_pt, ['down', 'up']
    )
    np.testing.assert_equal(
        binary_rv_nondef.values,
        np.array(['down', 'up']).astype('U')
    )

    with pytest.raises(ValueError):
        BayesianNodeRV('X0', binary_pt, ['b', 'a'])

    with pytest.raises(ValueError):
        BayesianNodeRV('X0', binary_pt, values=['a', 'a'])


def test_bnrv_equality(binary_pt, binary_cpt):
    rv = BayesianNodeRV('X0', binary_pt)
    assert rv == rv

    assert rv != BayesianNodeRV('X1', binary_pt)

    assert rv != BayesianNodeRV('X0', binary_pt, values=['down', 'up'])

    assert rv != BayesianNodeRV('X1', binary_cpt, parent_names=['X0'])


def test_sample_value():
    assert SampleValue.possible_default_value(1)
    assert not SampleValue.possible_default_value(-1)
    assert not SampleValue.possible_default_value(1.)
    assert not SampleValue.possible_default_value('alice')

    # Check instantiaion
    SampleValue(1)

    le = LabelEncoder()
    le.fit(['alice'])
    SampleValue('alice', label_encoder=le)

    # Passing value not in label encoder classes should raise an error
    with pytest.raises(ValueError):
        SampleValue('bob', label_encoder=le)


def test_get_probability_table(binary_cpt):
    rv1c0 = BayesianNodeRV('X1', binary_cpt, parent_names=['X0'])
    np.testing.assert_equal(
        rv1c0.get_probability_table(parent_values={'X0': SampleValue(1)}),
        binary_cpt[1, :]
    )

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
    rv2c01 = BayesianNodeRV('X2', pt_X2cX0X1, parent_names=['X0', 'X1'])
    np.testing.assert_equal(
        rv2c01.get_probability_table(
            parent_values={'X0': SampleValue(0), 'X1': SampleValue(1)}
        ),
        pt_X2cX0X1[0, 1, :]
    )


def test_get_pmf(binary_pt):
    rv = BayesianNodeRV('X0', binary_pt)
    assert rv.pmf(0) == binary_pt[0]
    assert rv.pmf(1) == binary_pt[1]

    with pytest.raises(ValueError):
        rv.pmf(2)

    with pytest.raises(ValueError):
        rv.pmf('alice')

    with pytest.raises(ValueError):
        rv.pmf(1, parent_values={'Z': SampleValue(0)})


def test_get_pmf_w_parents():
    rv = BayesianNodeRV(
        'Y',
        np.array([
            [0.2, 0.8],
            [0.7, 0.3]
        ]),
        parent_names=['X']
    )
    assert rv.pmf(0, parent_values={'X': SampleValue(1)}) == 0.7

    with pytest.raises(ValueError):
        rv.pmf(1)

    le = LabelEncoder()
    le.fit(['alice', 'bob'])
    assert rv.pmf(
        1,
        parent_values={'X': SampleValue('alice', label_encoder=le)}
    ) == 0.8

    with pytest.raises(ValueError):
        rv.pmf(1, parent_values={'X': SampleValue('terry', label_encoder=le)})


def test_rvs(binary_pt, binary_cpt):
    rv = BayesianNodeRV('X0', binary_pt)
    assert isinstance(rv.rvs(seed=42)[0], np.int64)

    rv1c0 = BayesianNodeRV('X1', binary_cpt, parent_names=['X0'])
    draw = rv1c0.rvs(parent_values={'X0': SampleValue(1)}, seed=42)[0]
    assert draw in rv1c0.values

    draws = rv1c0.rvs(size=10, parent_values={'X0': SampleValue(1)}, seed=42)
    assert len(draws) == 10

    # Test handling of extraneous parent values, needed for ancestral sampling
    assert (
        rv1c0.rvs(parent_values={'X0': SampleValue(1)}, seed=42)[0] ==
        rv1c0.rvs(
            parent_values={'X0': SampleValue(1), 'X2': SampleValue(42)}, seed=42
        )[0]
    )

    # Test non-default value sampling
    rv_nondef = BayesianNodeRV('X0', binary_pt, values=['down', 'up'])
    assert rv_nondef.rvs(seed=42)[0] in rv_nondef.values
