import numpy as np
import pytest

from fake_data_for_learning.probability_polytopes import (
    get_simplex_sample, ExpectationConstraint, ProbabilityPolytope,
    MapMultidimIndexToLinear
)


class TestGetSimplexSample:
    def test_get_unit_box_sample(self):
        sample = get_simplex_sample(3)

        # length of sample correct
        assert len(sample) == 3

        # sample >= 0
        assert np.all(sample >= 0)

        # sample <= 1
        assert np.all(sample <= 1)

        # sample sums to 1
        assert sample.sum() == pytest.approx(1.)


class TestProbabilityPolytope:

    def test_init(self):
        # dimensions must be an iterable
        with pytest.raises(ValueError):
            ProbabilityPolytope(('outcome'), dict(outcome=range(2)))

    polytope = ProbabilityPolytope(
        dims=('input', 'more_input', 'output'),
        coords=dict(input=['hi', 'low'], more_input=range(2), output=range(2))
    )

    def test_get_sum_dims(self):
        self.polytope.get_sum_dims(
            dict(input='low')
        ) == ('more_input', 'output')

    def test_get_sum_overs(self):
        expected = [
            dict(more_input=0, output=0),
            dict(more_input=0, output=1),
            dict(more_input=1, output=0),
            dict(more_input=1, output=1)
        ]
        assert self.polytope.get_sum_overs(
            dict(input='low')
        ) == expected

    def test_get_n_outcomes(self):
        assert self.polytope.get_n_outcomes() == 2*2*2

    def test_get_n_probability_constraints(self):

        tertiary = ProbabilityPolytope(dims=('v',), coords=dict(v=range(3)))
        assert isinstance(tertiary.get_n_probability_constraints(), int)

        assert self.polytope.get_n_probability_constraints() \
            == 2*2

    def test_get_total_probability_constraint_equation(self):
        assert list(self.polytope.get_total_probability_constraint_equations()) == \
            [
                dict(input='hi', more_input=0),
                dict(input='hi', more_input=1),
                dict(input='low', more_input=0),
                dict(input='low', more_input=1)
            ]

    def test_get_total_probability_constraint_matrix(self):
        np.testing.assert_array_almost_equal(
            self.polytope.get_total_probability_constraint_matrix(),
            np.array([
                [1., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1.],
            ])
        )

    def test_get_probability_bounds_half_planes(self):
        A_expected = np.vstack([np.eye(2*2*2), -np.eye(2*2*2)])
        b_expected = np.hstack([np.ones(2*2*2), np.zeros(2*2*2)])

        A, b = self.polytope.get_probability_bounds_half_planes()

        np.testing.assert_array_almost_equal(A_expected, A)
        np.testing.assert_array_almost_equal(b_expected, b)

    def test_get_half_planes_from_equations(self):

        A = np.array([
            [1., 0.],
        ])
        b = np.array([1])

        Ap, bp = ProbabilityPolytope.get_half_planes_from_equations(A, b)

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

    def test_get_probability_half_planes(self):
        # Test array sizes
        A, b = self.polytope.get_probability_half_planes()

        assert A.shape[0] == len(b)

        # Test results on smaller example
        small_polytope = ProbabilityPolytope(
                ('input', 'output'),
                dict(input=['hi', 'low'], output=range(2))
            )
        A_expect = np.vstack([
            # Total probability
            np.array([
                [1., 1., 0., 0.],
                [0., 0., 1., 1.],
                [-1., -1., 0., 0.],
                [0., 0., -1., -1.]
            ]),
            # 0 <= p <= 1
            np.eye(4),
            -1 * np.eye(4),
        ])

        b_expect = np.hstack([
            np.array([1., 1., -1., -1.]),
            np.ones(4), np.zeros(4),
        ])

        A, b = small_polytope.get_probability_half_planes()

        np.testing.assert_array_almost_equal(A, A_expect)
        np.testing.assert_array_almost_equal(b, b_expect)


class TestAddPolytopeConstraints:
    constrained_polytope = ProbabilityPolytope(
            ('input', 'output'),
            dict(input=['hi', 'low'], output=range(2))
        )

    def test_add_expectation_constraint(self):
        # ValueError: Invalid expectation value constraint
        with pytest.raises(ValueError):
            self.constrained_polytope.set_expectation_constraints(
                [ExpectationConstraint(equation=dict(input=0), moment=1, value=0.5)]
            )

        # ValueError: Expectation constraint on dimension over which expectation
        # value is calculated
        with pytest.raises(ValueError):
            self.constrained_polytope.set_expectation_constraints(
                [ExpectationConstraint(equation=dict(output=0), moment=1, value=42)],
            )

    # Add constraint
    constrained_polytope.set_expectation_constraints(
        [ExpectationConstraint(equation=dict(input='low'), moment=1, value=0.5)]
    )

    # Conditional probability with two inputs
    two_input_constrained_polytope = ProbabilityPolytope(
        dims=('input', 'more_input', 'output'),
        coords=dict(input=['hi', 'low'], more_input=range(2), output=range(2))
    )
    two_input_constrained_polytope.set_expectation_constraints(
        [ExpectationConstraint(equation=dict(more_input=0), moment=1, value=0.5)]
    )

    def test_get_expect_eq_col_indices(self):
        assert self.constrained_polytope.get_expect_equations_col_indices(
            dict(input='low')
        ) == [2, 3]

        assert self.two_input_constrained_polytope.get_expect_equations_col_indices(
            dict(more_input=0)
        ) == [0, 1, 4, 5]

    def test_get_expect_equation_coefficient(self):
        assert self.constrained_polytope.get_expect_equation_coefficient(
            dict(input='low')
        ) == pytest.approx(1.)
        assert self.two_input_constrained_polytope.get_expect_equation_coefficient(
            dict(input='low')
        ) == pytest.approx(1 / 2.)

    def test_get_expect_equations_matrix(self):
        np.testing.assert_almost_equal(
            self.constrained_polytope.get_expect_equations_matrix(),
            np.array([[0, 0, 0, 1]])
        )

        np.testing.assert_array_almost_equal(
            self.two_input_constrained_polytope.get_expect_equations_matrix(),
            1 / 2. * np.array([[0., 1., 0., 0., 0., 1., 0., 0.]])
        )

    def test_get_all_halfplanes(self):
        A_expect = np.vstack([
            # Total probability
            np.array([
                [1., 1., 0., 0.],
                [0., 0., 1., 1.],
                [-1., -1., 0., 0.],
                [0., 0., -1., -1.]
            ]),
            # 0 <= p <= 1
            np.eye(4),
            -1 * np.eye(4),
            # Expectation constraint
            np.array([
                [0., 0., 0., 1.],
                [0., 0., 0., -1.]
            ])
        ])

        b_expect = np.hstack([
            np.array([1., 1., -1., -1.]),
            np.ones(4), np.zeros(4),
            np.array([0.5, -0.5])
        ])
        A, b = self.constrained_polytope.get_all_half_planes()

        np.testing.assert_array_almost_equal(A, A_expect)
        np.testing.assert_array_almost_equal(b, b_expect)


class TestPolytopeVertexRepresentation:
    bernoulli = ProbabilityPolytope(('outcome',), dict(outcome=range(2)))
    conditional_bernoullis = ProbabilityPolytope(
        ('input', 'output'), dict(input=['hi', 'low'], output=range(2))
    )

    def test_get_vertex_representation(self):
        np.testing.assert_array_almost_equal(
            self.bernoulli.get_vertex_representation(),
            np.array([
                [1., 0.],
                [0., 1.]
            ])
        )

        np.testing.assert_array_almost_equal(
            self.conditional_bernoullis.get_vertex_representation(),
            np.array([
                [1., 1., 0., 0.],
                [0., 0., 1., 1.],
                [1., 0., 0., 1.],
                [0., 1., 1., 0.]
            ])
        )

        # Add expectation constraint
        self.conditional_bernoullis.set_expectation_constraints(
            [ExpectationConstraint(equation=dict(input='low'), moment=1, value=0.5)]
        )

        # Ensure dimension of vertices (column vectors) match ambient space
        assert self.conditional_bernoullis.get_vertex_representation().shape[0] == 4

    def test_generate_flat_random_cpt(self):
        flat_cpt = self.conditional_bernoullis.generate_flat_random_cpt()

        # flat cpt must contain same number of entries as possible outcomes
        assert flat_cpt.shape[0] == self.conditional_bernoullis.get_n_outcomes()

        idx_conditioned = [
            self.conditional_bernoullis.map_multidim_to_linear.to_linear(('low', 0)),
            self.conditional_bernoullis.map_multidim_to_linear.to_linear(('low', 1))
        ]
        np.testing.assert_array_almost_equal(
            flat_cpt[idx_conditioned],
            np.array([0.5, 0.5])

        )

    def test_get_random_cpt(self):
        cpt = self.conditional_bernoullis.generate_random_cpt()

        assert cpt.shape == (2, 2)

        # Test probabilities sum to 1.
        assert cpt[0, :].sum() == pytest.approx(1.)
        assert cpt[1, :].sum() == pytest.approx(1.)

        # Test input=1 expectation value constraint
        assert cpt[1, 0] == pytest.approx(0.5)
        assert cpt[1, 1] == pytest.approx(0.5)


class TestMultidimIndexToLinear:
    # Test instantiation
    with pytest.raises(ValueError):
        MapMultidimIndexToLinear(('outcome'), dict(outcome=range(2)))

    multi_to_linear = MapMultidimIndexToLinear(
        ('input', 'output'),
        dict(input=['hi', 'low'], output=range(3))
    )

    def test_dim(self):
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
