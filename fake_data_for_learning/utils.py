"""Utility functions for fake_data_for_learning"""
from itertools import product
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import xarray as xr

from scipy.special import softmax

from pypoman import compute_polytope_vertices


class RandomCpt:
    """Generate random conditional probability table"""
    def __init__(self, *shape):
        self.shape = shape

    def __call__(self, seed=None):
        """
        Generate non-negative random matrix of given shape such that sums over
        last dimension are 1.
        """
        if seed is not None:
            np.random.seed(seed=seed)
        res = np.random.rand(*self.shape)
        res = self.make_cpt(res)
        return res

    @staticmethod
    def make_cpt(x):
        """
        Convert numpy array x to conditional probability table.
        """
        res = x.copy()
        ranges = [range(s) for s in res.shape[:-1]]
        for s in product(*ranges):
            res[s] = softmax(res[s])

        return res


def get_simplex_sample(ambient_dimension, size=None):
    """
    Get random element of the simplex of given ambient dimension

    Parameters
    ----------
    ambient_dimension : int
        Dimension of ambient real vector space in which (probability) simplex
        is defined.

    Returns
    -------
    res : np.array
        Random sample point from the simplex
    """
    res = np.random.dirichlet(np.ones(ambient_dimension), size=size)
    return res


ExpectationConstraint = namedtuple(
    'ExpectationConstraint', ['equation', 'moment', 'value']
)


class ProbabilityPolytope:
    """
    Polytope represenation and methods for discrete (conditional) probability
    distributions.

    Parameters
    ----------
     dims : iterable of strings
        Dimension names

     coords : dict
        Dict with keys dimension names and values dimension values
    """
    def __init__(self, dims=tuple(), coords=dict()):
        self.dims = dims
        self.expect_on_dimension = dims[-1]
        self.map_multidim_to_linear = MapMultidimIndexToLinear(dims, coords)
        self.coords = coords
        self.expect_constraints = []

    def get_probability_half_planes(self):
        """
        Get half plane representations of polytope defined by
            * the probability polytope half planes due to
                + probabilities summing to one, and
                + probabilities lying in [0, 1], plus
            * the given expectation value constraints

        Returns
        -------
        A, b : tuple of np.array
        """
        A_total_prob, b_total_prob = self.get_total_probability_half_planes()
        A_prob_bounds, b_prob_bounds = \
            self.get_probability_bounds_half_planes()

        return np.vstack([A_total_prob, A_prob_bounds]), \
            np.hstack([b_total_prob, b_prob_bounds])

    def get_total_probability_half_planes(self):
        """
        Returns
        -------
        A, b : tuple of np.array
        """
        # Equality representation
        A = self.get_total_probability_constraint_matrix()
        b = np.ones(self.get_n_probability_constraints())

        # Half-plane representation
        A, b = self.get_half_planes_from_equations(A, b)

        return A, b

    def get_total_probability_constraint_matrix(self):
        """
        Returns
        -------
        A : np.array
        """
        probability_constraint_equations = \
            self.get_total_probability_constraint_equations()

        A = np.zeros((
            self.get_n_probability_constraints(),
            self.map_multidim_to_linear.dim
        ))
        for idx, equation in enumerate(probability_constraint_equations):
            cols = self.get_expect_equations_col_indices(equation)
            A[idx, cols] = self.get_expect_equations_row_entries(equation, 0)

        return A

    def get_n_probability_constraints(self):
        """
        Get the number of probability constraints involved in calculating
        expectation values over self.expect_on_dimension.

        Returns
        -------
        res : int
        """
        res = int(
            self.get_n_outcomes() / len(self.coords[self.expect_on_dimension])
        )
        return res

    def get_n_outcomes(self):
        dim_cards = [len(self.coords[d]) for d in self.dims]
        res = int(np.prod(dim_cards))

        return res

    def get_total_probability_constraint_equations(self):
        """
        Get iterator of constraint equation for conditional probabilities
        summing to 1.

        Returns
        -------
        : itertools.product
        """
        res = self.coords.copy()
        res.pop(self.expect_on_dimension)

        return product_dict(**res)

    def get_probability_bounds_half_planes(self):
        """
        Get half-plane representation of bounds 0 <= p_I <= 1 for all possible
        value indices I.

        Returns
        -------
        A, b : tuple of np.array
        """
        A = np.vstack([
            np.eye(self.map_multidim_to_linear.dim),
            -np.eye(self.map_multidim_to_linear.dim)
        ])
        b = np.hstack([
            np.ones(self.map_multidim_to_linear.dim),
            np.zeros(self.map_multidim_to_linear.dim)
        ])

        return A, b

    @staticmethod
    def get_half_planes_from_equations(A, b):
        """
        Convert equations of the form Ax = b to inequalties of the form
        A' x' <= b'

        Parameters
        ----------
        A : numpy.array of shape (m,n)
        b : numpy.array of shape (n,)

        Returns
        -------
        Ap : numpy.array of shape (2m, n)
        bp : numpy.array of shape (2n,)
        """
        Ap = np.vstack([A, -1*A])
        bp = np.hstack([b, -1*b])

        return Ap, bp

    def set_expectation_constraints(self, constraints):
        self._validate_expectation_constraint(constraints)
        self.expect_constraints = constraints

    def _validate_expectation_constraint(self, constraints):
        if self.expect_on_dimension in flatten_list(
            [c.equation.keys() for c in constraints]
        ):
            msg = (
                    f"Final array dimension {self.expect_on_dimension} "
                    "reserved for expectation, and may not be included in "
                    "expect_constraints"
                )
            raise ValueError(msg)

        invalid_equations = []
        for constraint in constraints:
            equation = constraint.equation
            for k, v in equation.items():
                if v not in self.coords[k]:
                    invalid_equations.append(equation)
        if invalid_equations:
            msg = f"Invalid constraint equations {invalid_equations}"
            raise ValueError(msg)

    def get_all_half_planes(self):
        """
        Get half plane representations of polytope defined by
            * the probability polytope half planes due to
                + probabilities summing to one, and
                + probabilities lying in [0, 1], plus
            * the given expectation value constraints

        Returns
        -------
        A, b : tuple of np.array
        """
        As = []
        bs = []
        A_prob, b_prob = self.get_probability_half_planes()
        As.append(A_prob)
        bs.append(b_prob)

        if self.expect_constraints:
            A_expect, b_expect = self.get_expect_equations_half_planes()
            As.append(A_expect)
            bs.append(b_expect)

        return np.vstack(As), np.hstack(bs)

    def get_expect_equations_half_planes(self):
        A_equations = self.get_expect_equations_matrix()
        b_equations = np.array(
            [constraint.value for constraint in self.expect_constraints]
        )

        A_half_plane, b_half_plane = self.get_half_planes_from_equations(
            A_equations, b_equations
        )

        return A_half_plane, b_half_plane

    def get_expect_equations_matrix(self):
        """
        Generate (linearly indexed) equation matrix from expect_constraints

        Returns
        -------
        A : numpy.array
        """
        A = np.zeros(
            (len(self.expect_constraints), self.map_multidim_to_linear.dim)
        )
        for idx, constraint in enumerate(self.expect_constraints):
            cols = self.get_expect_equations_col_indices(constraint.equation)
            A[idx, cols] = self.get_expect_equations_row_entries(
                constraint.equation, constraint.moment
            )

            A[idx, :] = self.get_expect_equation_coefficient(
                constraint.equation
            ) * A[idx, :]
        return A

    def get_expect_equations_row_entries(self, constraint_equation, moment):
        """
        Parameters
        ----------
        constraint_equation : dict

        Returns
        -------
        : list of int
            Values of self.expect_on_dimension in summand
        """
        sum_overs = self.get_sum_overs(constraint_equation)
        return [
            sum_over[self.expect_on_dimension] ** moment
            for sum_over in sum_overs
        ]

    def get_expect_equation_coefficient(self, constraint_equation):
        """
        Parameters
        ----------
        constraint_equation : dict

        Returns
        -------
        res : float
        """
        sum_dims = self.get_sum_dims(constraint_equation)
        normalization_dims = list(set(sum_dims) - {self.expect_on_dimension})
        denom = 1
        for d in normalization_dims:
            denom *= len(self.coords[d])

        return 1 / float(denom)

    def get_expect_equations_col_indices(self, constraint_equation):
        sum_overs = self.get_sum_overs(constraint_equation)
        cols = [self.map_multidim_to_linear.to_linear(
            self.map_multidim_to_linear.get_coord_tuple(
                {**constraint_equation, **sum_over}
            )
        ) for sum_over in sum_overs]
        return cols

    def get_sum_overs(self, constraint_equation):
        """
        Parameters
        ----------
        constraint_equation : dict

        Returns
        -------
        : list
        """
        sum_dims = self.get_sum_dims(constraint_equation)
        sum_ranges = {
            dim: self.map_multidim_to_linear.coords[dim] for dim in sum_dims
        }

        return list(product_dict(**sum_ranges))

    def get_sum_dims(self, constraint_equation):
        """
        Parameters
        ----------
        constraint_equation : dict

        Returns
        -------
        res : tuple
            Matrix dimension names not among constraint equations
        """
        res = tuple(
            [d for d in self.dims if d not in constraint_equation.keys()]
        )
        return res

    def generate_random_cpt(self):
        """
        Generate random (conditional) probability table from the polytope
        """
        res = self._initialize_cpt()

        flat_cpt = self.generate_flat_random_cpt()
        for flat_idx in range(flat_cpt.shape[0]):
            multi_idx = self.map_multidim_to_linear.to_multidim(flat_idx)
            res.loc[multi_idx] = flat_cpt[flat_idx]

        return res.data

    def _initialize_cpt(self):
        coord_values = [self.coords[d] for d in self.dims]
        shape = [len(coord_value) for coord_value in coord_values]
        data = np.zeros(shape=shape)

        return xr.DataArray(data, coords=self.coords, dims=self.dims)

    def generate_flat_random_cpt(self):
        """
        Generate flat (i.e. array with only 1 dimension) representation of a
        random (conditional) probability table from the polytope.

        Returns
        -------
        res : np.array
            Float array of shape (number of possible outcomes, )

        """
        V = self.get_vertex_representation()
        t = get_simplex_sample(V.shape[1])
        res = np.matmul(V, t)
        return res

    def get_vertex_representation(self):
        """
        Calculate the vertex representation of the probability polytope.

        Returns
        -------
        res : np.array
            Matrix of shape self.dim, total number of constraints
        """
        A, b = self.get_all_half_planes()
        verts = compute_polytope_vertices(A, b)
        Vt = np.vstack(verts)
        V = np.transpose(Vt)
        return V


class MapMultidimIndexToLinear:
    """Convert multidimensional array indices to linear and back"""
    def __init__(self, dims, coords):
        self.dims = self._get_dims(dims)
        self.coords = self._get_coords(coords)
        self.mapping = self._get_mapping()
        self.dim = self._get_dim()

    def _get_dims(self, dims):
        """
        Perform check(s) and type-fixing for dims

        Parameters
        ----------
        dims : iterable
        """
        if isinstance(dims, str):
            raise(ValueError(f'dims must be an iterable, you entered {dims}'))

        return dims

    def _get_coords(self, coords):
        """Enforce ordering from self.dims"""
        return OrderedDict([(k, coords[k]) for k in self.dims])

    def _get_mapping(self):
        flat_coord_values = list(product(*self.coords.values()))
        return pd.Series(
            range(len(flat_coord_values)), index=flat_coord_values
        )

    def _get_dim(self):
        """Calcuate dimension of space of array entries"""
        return len(self.mapping.index)

    def to_linear(self, multi_index_value):
        return self.mapping[multi_index_value]

    def to_multidim(self, linear_index_value):
        return self.mapping.index[linear_index_value]

    def get_coord_tuple(self, coord_index_dict):
        pre_tuple = []
        for d in self.dims:
            pre_tuple.append(coord_index_dict[d])
        return tuple(pre_tuple)


def product_dict(**kwargs):
    """
    From StackOverflow: 5228158/cartesian-product-of-a-dictionary-of-lists
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def flatten_list(l):
    """
    From StackOverflow: 952914/how-to-make-a-flat-list-out-of-list-of-lists
    """
    return [item for sublist in l for item in sublist]
