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



multi_to_linear = ut.MultidimIndexToLinearMapping(['hi', 'low'], range(3))

def test_to_linear():
    multi_to_linear.to_linear(('hi', 0)) == 0
    multi_to_linear.to_linear(('low', 2)) == 5

def test_to_multidim():
    multi_to_linear.to_multidim(0) == ('hi', 0)
    multi_to_linear.to_multidim(5) == ('low', 2)