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


class TestConditionalProbabilityConstrainExpectation:
    
    def test_init(self):
        with pytest.raises(ValueError):
            ut.ConditionalProbabilityConstrainExpectation(
                [ut.ExpectationConstraint(equation=dict(output=0), value=42)],
                ('input', 'output'),
                dict(input=['hi', 'low'], output=range(3))
            )

    constrain_expectation = ut.ConditionalProbabilityConstrainExpectation(
        [ut.ExpectationConstraint(equation=dict(input='low'), value=42)],
        ('input', 'more_input', 'output'),
        dict(input=['hi', 'low'], more_input=range(2), output=range(2))
    )

    def test_get_sum_dims(self):
        self.constrain_expectation.get_sum_dims(
            dict(input='low')
        ) == ('more_input', 'output')

    def test_get_sum_overs(self):
        expected = [
            dict(more_input=0, output=0), 
            dict(more_input=0, output=1), 
            dict(more_input=1, output=0),
            dict(more_input=1, output=1)
        ]
        assert self.constrain_expectation.get_sum_overs(
            dict(input='low')
        ) == expected

    def test_get_expect_eq_col_indices(self):
        assert self.constrain_expectation.get_expect_equations_col_indices(
            dict(input='low')
        ) == [4, 5, 6, 7]

    def test_get_expect_equations_matrix(self):
        # Oth moment expectation
        np.testing.assert_almost_equal(
            self.constrain_expectation.get_expect_equations_matrix(0),
            np.array([[0., 0., 0., 0., 1., 1., 1., 1.]])
        )

        # First moment expectation
        np.testing.assert_almost_equal(
            self.constrain_expectation.get_expect_equations_matrix(1),
            np.array([[0., 0., 0., 0., 0., 1., 0., 1.]])
        )

    def test_get_expect_equation_coefficient(self):
        self.constrain_expectation.get_expect_equation_coefficient(
            dict(input='low')
        ) == pytest.approx(1 / 2.)

        small_expectation = ut.ConditionalProbabilityConstrainExpectation(
                [ut.ExpectationConstraint(equation=dict(input=0), value=3)],
                ('input', 'output'),
                dict(input=['hi', 'low'], output=range(3))
            )
        small_expectation.get_expect_equation_coefficient(
            dict(input='low')
        ) == pytest.approx(1.)

    def test_get_n_probability_constraints(self):
        assert self.constrain_expectation.get_n_probability_constraints() \
            == 2*2

    def test_get_total_probability_constraint_equation(self):
        assert list(self.constrain_expectation.get_total_probability_constraint_equations()) == \
            [
                dict(input='hi', more_input=0),
                dict(input='hi', more_input=1),
                dict(input='low', more_input=0),
                dict(input='low', more_input=1)
            ]

    def test_get_total_probability_constraint_matrix(self):
        print(self.constrain_expectation.get_total_probability_constraint_matrix())
        np.testing.assert_array_almost_equal(
            self.constrain_expectation.get_total_probability_constraint_matrix(),
            np.array([
                [1., 1., 0., 0., 0., 0., 0., 0.], 
                [0., 0., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1.],
            ])
        )
    
    def test_get_probability_bounds_half_planes(self):
        A_expected = np.concatenate([np.eye(2*2*2), -np.eye(2*2*2)], axis=0)
        b_expected = np.concatenate([np.ones(2*2*2), np.zeros(2*2*2)], axis=0)
        
        A, b = self.constrain_expectation.get_probability_bounds_half_planes()
        
        np.testing.assert_array_almost_equal(A_expected, A)
        np.testing.assert_array_almost_equal(b_expected, b)

    def test_get_half_planes_from_equations(self):

        A = np.array([
            [1., 0.],
        ])
        b = np.array([1])

        Ap, bp = ut.ConditionalProbabilityConstrainExpectation.get_half_planes_from_equations(A, b)

        np.testing.assert_array_almost_equal(
            Ap,
            np.array([
                [1., 0.],
                [-1., 0.]
            ])
        )

        np.testing.assert_array_almost_equal(
            bp,
            np.array([1, -1])
        )

    def test_get_all_half_planes(self):
        A, b = self.constrain_expectation.get_all_half_planes()

        assert A.shape[0] == len(b)


class TestMultidimIndexToLinear:
    multi_to_linear = ut.MapMultidimIndexToLinear(
        ('input', 'output'),
        dict(input=['hi', 'low'], output=range(3))
    )

    def test_dim(self):
        print(self.multi_to_linear.dim)
        assert self.multi_to_linear.dim == 6

    def test_to_linear(self):
        self.multi_to_linear.to_linear(('hi', 0)) == 0
        self.multi_to_linear.to_linear(('hi', 1)) == 1
        self.multi_to_linear.to_linear(('low', 2)) == 5

    def test_to_multidim(self):
        self.multi_to_linear.to_multidim(0) == ('hi', 0)
        self.multi_to_linear.to_multidim(5) == ('low', 2)

    @pytest.mark.parametrize(
        "coord_index_dict,expected", 
        [   
            (dict(output=0, input='hi'), ('hi', 0)),
            (dict(output=2, input='low'), ('low', 2))
        ]
    )
    def test_get_coord_tuple(self, coord_index_dict, expected):
        assert self.multi_to_linear.get_coord_tuple(coord_index_dict) == expected

