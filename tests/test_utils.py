import pytest

import numpy as np
from itertools import product

from fake_data_for_learning import utils as ut

@pytest.mark.parametrize(
    "test_input", 
    [   
        np.array([-1., 1.]),
        np.array([
            [1., 0.],
            [-1., 1.]]),
        np.ones((2,3,4))

    ]
)
def test_make_cpt(test_input):
    cpt = ut.RandomCpt.make_cpt(test_input)

    assert (cpt >= 0).all()

    # test that final dimensions sum to 1
    ranges = [range(s) for s in cpt.shape[:-1]]
    for r in product(*ranges):
        np.testing.assert_almost_equal(sum(cpt[r]), 1)


class TestConditionalProbabilityLinearConstraints:
    
    def test_init(self):
        with pytest.raises(ValueError):
            ut.ConditionalProbabilityLinearConstraints(
                [dict(output=0)],
                ('input', 'output'),
                dict(input=['hi', 'low'], output=range(3))
            )

    linear_constraints = ut.ConditionalProbabilityLinearConstraints(
        [dict(input='low')],
        ('input', 'output'),
        dict(input=['hi', 'low'], output=range(3))
    )

    def test_get_sum_dims(self):
        self.linear_constraints.get_sum_dims(0) == ('output')

    def test_get_sum_overs(self):
        expected = [dict(output=0), dict(output=1), dict(output=2)]
        assert self.linear_constraints.get_sum_overs(0) == \
            expected

    def test_get_lin_eq_col_indices(self):
        assert self.linear_constraints.get_lin_equations_col_indices(
            0
        ) == [3, 4, 5]



multi_to_linear = ut.MapMultidimIndexToLinear(
    ('input', 'output'),
    dict(input=['hi', 'low'], output=range(3))
)

def test_dim():
    print(multi_to_linear.dim)
    assert multi_to_linear.dim == 6

def test_to_linear():
    multi_to_linear.to_linear(('hi', 0)) == 0
    multi_to_linear.to_linear(('low', 2)) == 5

def test_to_multidim():
    multi_to_linear.to_multidim(0) == ('hi', 0)
    multi_to_linear.to_multidim(5) == ('low', 2)

@pytest.mark.parametrize(
    "coord_index_dict,expected", 
    [   
        (dict(output=0, input='hi'), ('hi', 0)),
        (dict(output=2, input='low'), ('low', 2))
    ]
)
def test_get_coord_tuple(coord_index_dict, expected):
    assert multi_to_linear.get_coord_tuple(coord_index_dict) == expected

