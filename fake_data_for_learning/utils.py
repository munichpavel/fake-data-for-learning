'''Utility functions for fake_data_for_learning'''
import string
from itertools import product
from collections import OrderedDict

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

class ConditionalProbabilityLinearConstraints:
    def __init__(self, constraints, dims, coords):
        """
        Expectation value of last element in dims subject to constraints
        """
        self.constraints = constraints
        self.dims = dims
        self.expect_on_dimension = dims[-1]
        self.map_multidim_to_linear = MapMultidimIndexToLinear(dims, coords)
        self.coords = coords
        self._validate()

    def _validate(self):
        if self.expect_on_dimension in flatten_list([c.keys() for c in self.constraints]):
            msg = f"Final array dimension {self.expect_on_dimension} is reserved " \
                "for expectation, and may not be included in constraints"
            raise ValueError(msg)

    def get_lin_equations_col_indices(self, constraint_index):
        sum_overs = self.get_sum_overs(constraint_index)
        cols = [self.map_multidim_to_linear.to_linear(
            self.map_multidim_to_linear.get_coord_tuple(
                {**self.constraints[constraint_index], **sum_over}
            )
        ) for sum_over in sum_overs]
        return cols

    def get_sum_overs(self, constraint_index):
        """
        Parameters
        ----------
        constraint_index : int
            Index of element of self.constraints

        Returns
        -------
        : list
        """
        sum_dims = self.get_sum_dims(constraint_index)
        sum_ranges = {dim: self.map_multidim_to_linear.coords[dim] \
            for dim in sum_dims}
        return list(product_dict(**sum_ranges))

    def get_sum_dims(self, constraint_index):
        """
        Parameters
        ----------
        constraint_index : int
            Index of element of self.constraints

        Returns
        -------
        res : tuple
            Matrix dimension names not among constraint equations
        """
        res = tuple([d for d in self.dims if d not in self.constraints[constraint_index].keys()])
        return res

    def get_lin_equations_matrix(self):
        """
        Generate (linearly indexed) equation matrix from constraints
        
        Returns
        -------
        A : numpy.array
        """
        A = np.zeros((len(self.constraints), self.map_multidim_to_linear.dim))
        for idx in range(len(self.constraints)):
            cols = self.get_lin_equations_col_indices(idx)
            A[idx, cols] = self.get_lin_equations_row(idx)
        return A

    def get_lin_equations_row(self, constraint_index):
        """
        Parameters
        ----------
        constraint_index : int
            Index of element of self.constraints

        Returns
        -------
        : list of int
            Values of self.expect_on_dimension in summand
        """
        sum_overs = self.get_sum_overs(constraint_index)
        return [sum_over[self.expect_on_dimension] for sum_over in sum_overs]

    def get_lin_equation_coefficient(self, constraint_index):
        """
        Parameters
        ----------
        constraint_index : int
            Index of element of self.constraints

        Returns
        -------
        res : float
        """
        sum_dims = self.get_sum_dims(constraint_index)
        normalization_dims = list(set(sum_dims) - {self.expect_on_dimension})
        denom = 0
        for d in normalization_dims:
            denom *= len(self.coords[d])

        denom = max(1, denom)
        return 1/denom


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