'''Utility functions for fake_data_for_learning'''
import string
from itertools import product
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd

from scipy.special import softmax


class RandomCpt:
    """Generate random conditional probability table"""
    def __init__(self, *shape):
        self.shape = shape

    def __call__(self, seed=None):
        """
        Generate non-negative random matrix of given 
        shape such that sums over last dimension are 1
        """
        if seed is not None:
            np.random.seed(seed=seed)
        res = np.random.rand(*self.shape)
        res = self.make_cpt(res)
        return res

    @staticmethod
    def make_cpt(x):
        """
        Convert numpy array x to conditional
        probability table
        """
        res = x.copy()
        ranges = [range(s) for s in res.shape[:-1]]
        for s in product(*ranges):
            res[s] = softmax(res[s])

        return res


ExpectationConstraint = namedtuple('ExpectationConstraint', ['equation', 'value'])


class ConditionalProbabilityConstrainExpectation:
    """
    
    Parameters
    ----------
     expect_constraints : list of utils.ExpectationConstraint
        Expectation value constraints as list of namedtuples with keys
        `equation` and `value`

     dims : iterable of strings
        Dimension names

     coords : dict
        Dict with keys dimension names and values dimension values
    
    """
    def __init__(self, expect_constraints, dims, coords):
        """
        Expectation value of last element in dims subject to expect_constraints
        """
        self.expect_constraints = expect_constraints
        self.dims = dims
        self.expect_on_dimension = dims[-1]
        self.map_multidim_to_linear = MapMultidimIndexToLinear(dims, coords)
        self.coords = coords
        self._validate()

    def _validate(self):
        if self.expect_on_dimension in flatten_list([c.equation.keys() for c in self.expect_constraints]):
            msg = f"Final array dimension {self.expect_on_dimension} is reserved " \
                "for expectation, and may not be included in expect_constraints"
            raise ValueError(msg)

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
        sum_ranges = {dim: self.map_multidim_to_linear.coords[dim] \
            for dim in sum_dims}
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
        res = tuple([d for d in self.dims if d not in constraint_equation.keys()])
        return res

    def get_expect_equations_matrix(self, moment):
        """
        Generate (linearly indexed) equation matrix from expect_constraints
        
        Returns
        -------
        A : numpy.array
        """
        A = np.zeros((len(self.expect_constraints), self.map_multidim_to_linear.dim))
        for idx, constraint in enumerate(self.expect_constraints):
            cols = self.get_expect_equations_col_indices(constraint.equation)
            A[idx, cols] = self.get_expect_equations_row(constraint.equation, moment)
        return A

    def get_expect_equations_row(self, constraint_equation, moment):
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
        return [sum_over[self.expect_on_dimension] ** moment for sum_over in sum_overs]

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
        denom = 0
        for d in normalization_dims:
            denom *= len(self.coords[d])

        denom = max(1, denom)
        return 1 / denom

    def get_total_probability_constraint_matrix(self):
        n_probability_constraints = np.prod(
            [len(self.coords[d]) for d in self.dims if d != self.expect_on_dimension]
        )

        probability_constraint_equations = self.get_total_probability_constraint_equations()
        A = np.zeros((
            n_probability_constraints,
            self.map_multidim_to_linear.dim 
        ))
        for idx, equation in enumerate(probability_constraint_equations):
            cols = self.get_expect_equations_col_indices(equation)
            A[idx, cols] = self.get_expect_equations_row(equation, 0)

        return A

    def get_total_probability_constraint_equations(self):
        """
        Get iterator of constraints for conditional probabilities summing to 1.

        Returns
        -------
        : itertools.product
        """
        res = self.coords.copy()
        res.pop(self.expect_on_dimension)
        
        return product_dict(**res)

    @staticmethod
    def get_half_plane_rep_from_equations(A, b):
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
        Ap = np.concatenate([A, -A], axis=0)
        bp = np.concatenate([b, -b], axis=0)
        return (Ap, bp)




class MapMultidimIndexToLinear:
    """Convert multidimensional array indices to linear and back"""
    def __init__(self, dims, coords):
        self.dims = dims
        self.coords = self._get_coords(coords)
        self.mapping = self._get_mapping()
        self.dim = self._get_dim()

    def _get_coords(self, coords):
        """Enforce ordering from self.dims"""
        return OrderedDict([(k, coords[k]) for k in self.dims])
    
    def _get_mapping(self):
        flat_coord_values = list(product(*self.coords.values()))
        return pd.Series(range(len(flat_coord_values)), index=flat_coord_values)

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
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def flatten_list(l):
    """
    https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    """
    return [item for sublist in l for item in sublist]