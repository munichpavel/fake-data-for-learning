import pytest

import numpy as np
from itertools import product

from fake_data_for_learning.utils import RandomCpt


@pytest.mark.parametrize(
    "test_input",
    [
        np.array([-1., 1.]),
        np.array([
            [1., 0.],
            [-1., 1.]]),
        np.ones((2, 3, 4))

    ]
)
def test_make_cpt(test_input):
    cpt = RandomCpt.make_cpt(test_input)

    assert (cpt >= 0).all()

    # test that final dimensions sum to 1
    ranges = [range(s) for s in cpt.shape[:-1]]
    for r in product(*ranges):
        np.testing.assert_almost_equal(sum(cpt[r]), 1)
