import numpy as np
from itertools import product

from fake_data_for_learning import utils as ut


random_cpt = ut.RandomCpt(3, 2)

def test_generate_random_cpt():
    #cpt = ut.generate_random_cpt(3,2)
    cpt = random_cpt()
    # test that entries are non-negative
    assert np.all(cpt >= 0)


# TODO Parametrize test
def test_make_cpt():
    random_cpt = ut.RandomCpt(3,2,4)
    cpt = random_cpt.make_cpt(np.random.rand(3,2,4))

    # test that final dimensions sum to 1
    ranges = [range(s) for s in cpt.shape[:-1]]
    for r in product(*ranges):
        np.testing.assert_almost_equal(sum(cpt[r]), 1)

    
    random_cpt = ut.RandomCpt(4,2,3)
    cpt_from_negative = random_cpt.make_cpt(-np.ones((4,2,3)))

    # test that final dimensions sum to 1
    ranges = [range(s) for s in cpt_from_negative.shape[:-1]]
    for r in product(*ranges):
        np.testing.assert_almost_equal(sum(cpt_from_negative[r]), 1)
