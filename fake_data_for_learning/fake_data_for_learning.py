# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d

from . import utils as ut


class BayesianNodeRV:
    r'''
    Sample-able random variable corresponding to node of a discrete Bayesian network.

    Parameters
    ----------
    name : string
        Node variable name
    cpt: numpy.array
        (Conditional) probability array. NB: We depart from normal convention on the assignment of the 
        array dimensions. In this class, the node variable (aka dependent, not conditioned on) corresponds
        to the LAST componend of array indexing. E.g. For a 2-d array cpt, this means that the ROWS must sum to 1,
        not the columns, as is otherwise standard. This choice is to make cpt definition in numpy more human-readable.

    values: list, optional
        list of values random variable will take. Default is [0, cpt.shape[-1]), if values
        are given, use sklearn.preprocessing.LabelEncoder(). NOTE: the sklearn label encoder
        sorts values in lexicographic order. To ensure compatibility with the conditional 
        probability table, given values must also be in lexicographic order.

    parents: list, optional
        list of parent node random variable names. Default is None, i.e. no parents
    '''
    def __init__(self, name, cpt, values=None, parent_names=None):
        self.name = name
        self.cpt = cpt
        self.parent_names = self._set_parent_names(parent_names)
        self._set_values(cpt, values)

    def __eq__(self, other):
  
        if self.__class__ != other.__class__: 
            return False
        return (
            self.name == other.name and
            np.array_equal(self.cpt, other.cpt) and
            self.parent_names == other.parent_names and
            np.array_equal(self.values, other.values)
        )

    def _set_parent_names(self, parent_names):
        
        if not(parent_names is None or isinstance(parent_names, list)):
            raise TypeError('parent_names must be a list or None')

        val_error_msg = (
            'Number of parent names and conditional probability '
            'table are incompatible'
        )
        if parent_names is None:
            if 1 != len(self.cpt.shape):
                raise ValueError(val_error_msg)
        else:
            # Check compability of parent names with conditional proby table
            if len(parent_names) + 1 != len(self.cpt.shape):
                raise ValueError(val_error_msg)

        return parent_names

    def _set_values(self, cpt, values):
        if values is None:
            self.values = np.array(range(cpt.shape[-1]))
            self.label_encoder = None
        else:
            # Confirm that non-default values are in lexicographic order
            if not np.array_equal(
                np.array(values),
                np.unique(values)
            ):
                raise ValueError('Values must be unique and in lexicographic order')
            self.label_encoder = self._set_label_encoder(values)
            self.values = self.label_encoder.classes_

    def _set_label_encoder(self, values):
        le = LabelEncoder()
        le.fit(values)
        return le

    def rvs(self, parent_values=None, size=1, seed=None):
        r'''
        Generate random variates from the bayesian node.

        Parameters
        -----------
        parent_values : None or dict
            None if node is an orphan, else a dict with parent values
        size : int
            Number of random samples to draw
        seed : int
            Seed for numpy.random
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.
        '''
        np.random.seed(seed)

        if self.parent_names is None:
            return np.random.choice(self.values, size, p=self.cpt)
        else:
            res = np.random.choice(self.values, size, p=self.get_pt(parent_values))
            return res
    
    def get_pt(self, parent_values=None):
        r'''
        Get probability table.

        Parameters
        ----------
        parent_values: None, dict of ints of form {'node_name': int}, 
                or dict of dicts with entries 'node_name': {'value': int, 'label_encoder': fitted label encoder}
            Values of parent nodes to get relevant 1-d submatrix of the (conditional) probability table
        '''
        if parent_values is None:
            return self.cpt
        else:
            s = [slice(None)] * len(self.cpt.shape)
            for idx, p in enumerate(self.parent_names):
                parent_internal_value = self.get_internal_value(parent_values[p])
                print(parent_internal_value)
                s[idx] = parent_internal_value
            return self.cpt[tuple(s)]

    def get_internal_value(self, sample_value):
        '''
        Translate SampleValue representation to natural number consistent with 
        conditional probability table definition.

        Parameters
        ----------
        sample_value: SampleValue

        Returns
        -------
        res : int
            Internal (integer) representation of external value

        '''
        if sample_value.label_encoder is not None:
            return sample_value.label_encoder.transform([sample_value.value])[0]
        else:
            return sample_value.value

    def __repr__(self):
        return 'BayesianNodeRV({}, parent_names={})'.format(self.name, self.parent_names)


class SampleValue:
    def __init__(self, value, label_encoder=None):
        self.label_encoder = label_encoder
        self.value = self._set_value(value)

    def _set_value(self, value):
        if self.label_encoder is None:
            if SampleValue.possible_default_value(value):
                return value
            else:
                 raise ValueError('Non-default values require a label encoder')
        else:
            if not value in self.label_encoder.classes_:
                raise ValueError ('Value has not been encoded')
            else:
                return value

    @staticmethod
    def possible_default_value(x):
        r'''Check conditions that rule-out a default (i.e. natural number) value'''
        if isinstance(x, np.int) or isinstance(x, np.int64):
            return x >= 0    
        else:
            return False

    def __repr__(self):
        return 'SampleValue({}, {})'.format(self.value, self.label_encoder)


class FakeDataBayesianNetwork:
    r'''
    Sample-able Bayesian network comprised up of BayesianNetworkRV's

    Parameters
    -----------
    args : tuple of BayesianNetworkRV's (BNRVs)
        BayesianNetworkRV's that make up the Bayesian network

    (Other) Attributes
    ------------------
    node_names : list of strings
        Node variable names of the BNRVs
    adjacency_matrix : numpy array
        Adjanency matrix of the Bayesian network's graph
    _eve_node_names: list of strings
        Node variable names without parents
    '''
    def __init__(self, *args):
        self._bnrvs = args
        self.node_names = self._set_node_names()
        self.adjacency_matrix = self.calc_adjacency_matrix()
        self._eve_node_names = self._set_eve_node_names()
        self._validate_bn()
    
    def _set_node_names(self):
        node_names = []
        parent_names = []

        for rv in self._bnrvs:
            node_names.append(rv.name)
            if rv.parent_names is not None:
                parent_names += rv.parent_names

        missing_nodes = set(parent_names) - set(node_names)

        if missing_nodes != set():
            raise ValueError('Missing nodes from network: {}'.format(missing_nodes))
        
        return node_names

    def _validate_bn(self):
        r'''Check for consistency of node random variables in Bayesian network'''

        # Check consistency of conditional probability tables between parents and children
        for idx, rv in enumerate(self._bnrvs):
            parent_idxs = self.get_parent_idx(idx, self.adjacency_matrix)
            expected_cpt_dims = self.get_expected_cpt_dims(parent_idxs, len(rv.values))
        if rv.cpt.shape != tuple(expected_cpt_dims):
            raise ValueError(
                '{} conditional probability table dimensions {} inconsistent with parent values {}'.format(
                rv.name, rv.cpt.shape, expected_cpt_dims)
            )

    @staticmethod
    def get_parent_idx(child_idx, adjacency_matrix):
        r'''Return list of index positions of parents of node at child_idx in adjacency matrix'''
        res = np.nonzero(adjacency_matrix[:, child_idx])[0]
        return res.tolist()

    def get_expected_cpt_dims(self, parent_idxs, child_value_length):
        expected_cpt_dims = []
        for parent_idx in parent_idxs:
            expected_cpt_dims.append(len(self._bnrvs[parent_idx].values))

        # append node value length
        expected_cpt_dims.append(child_value_length)
        return expected_cpt_dims

    def calc_adjacency_matrix(self):
        
        res = np.zeros((len(self._bnrvs), len(self._bnrvs)), dtype=int)
        for i, node_i in enumerate(self._bnrvs):
            for j, node_j in enumerate(self._bnrvs):
                res[i,j] = FakeDataBayesianNetwork.name_in_list(node_i.name, node_j.parent_names)
        return res

    @staticmethod
    def name_in_list(name, l):
        r'''Return 1 if name is in list l, else 0'''
        if l is None:
                return 0
        res = name in l
        return res

    def _set_eve_node_names(self):
        r'''Find eve nodes as zero columns of adjacency matrix'''
        eve_idx = self.zero_column_idx(self.adjacency_matrix)
        res = list(np.array(self.node_names)[eve_idx])
        return res

    @staticmethod
    def zero_column_idx(X):
        r'''Return array with column indices of 0 columns'''
        return np.where(~X.any(axis=0))[0]

    def rvs(self, size=1, seed=None):
        r'''Ancestral sampling from the Bayesian network'''
        res = [self._rv_dict(seed=seed) for _ in range(size)]
        return pd.DataFrame.from_records(res, index=range(size), columns=self.node_names)

    def _rv_dict(self, seed=None):
        r'''Ancestral sampling from the Bayesian network'''
        samples_dict = {}
        sample_next_names = self._eve_node_names
        
        while not self.all_nodes_sampled(samples_dict):
            for node_name in sample_next_names:
                if self.all_parents_sampled(node_name, samples_dict):
                    node = self.get_node(node_name)
                    samples_dict[node_name] = SampleValue(node.rvs(parent_values=samples_dict, seed=seed)[0], node.label_encoder)
            #sample_next_names = self._get_sample_next_names(sample_next_names)
            sample_next_names = self.get_unsampled_nodes(samples_dict)
        # Keep only sample values
        return {k: v.value for (k,v) in samples_dict.items()}

    def all_nodes_sampled(self, samples_dict):
        return set(samples_dict.keys()) == set(self.node_names)

    def get_node(self, node_name):
        if node_name not in self.node_names:
            raise ValueError('No node defined with name {}'.format(node_name))
        res = self._bnrvs[self.node_names.index(node_name)]
        return res

    def all_parents_sampled(self, node_name, samples_dict):
        parent_names = self.get_node(node_name).parent_names
        if parent_names is None:
            return True
        sampled_names = set(samples_dict.keys())
        return set(parent_names).issubset(sampled_names)

    def get_unsampled_nodes(self, samples_dict):
        return list(set(self.node_names) - set(samples_dict.keys()))

    def _get_sample_next_names(self, current_names):
        idx_current_names = np.array([self.node_names.index(name) for name in current_names])
        idx_next_names = FakeDataBayesianNetwork.get_pure_descendent_idx(idx_current_names, self.adjacency_matrix)
        sample_next_names = [self.node_names[idx] for idx in idx_next_names]
        return sample_next_names

    @staticmethod
    def get_pure_descendent_idx(parent_idx, adjacency_matrix):
        r'''
        Return column ids of descendents having only parent_idx as parents.
        For parent indices i,j, returns k if and only if
        (
            adjacency_matrix[i,k] == adjacency_matrix[j,k] == 1
            and i,j are the only such indices (i.e.
                adjacency_matrix[i',k] == 1 implies i' \in {i,j})
        )
        '''
        n_vars = adjacency_matrix.shape[0]
        def mask_from_idx(idx):
            return np.array([x in idx for x in range(n_vars)])

        parent_mask = mask_from_idx(parent_idx)
        descendents = FakeDataBayesianNetwork.non_zero_column_idx(adjacency_matrix[parent_idx, :])

        # Keep only descendents having only parent_idx as parents
        pure_descendents = []
        for idx in descendents:
            if not adjacency_matrix[~parent_mask, idx].any():
                pure_descendents.append(idx)
        
        return np.array(pure_descendents)
        
    @staticmethod
    def non_zero_column_idx(X):
        r'''Return array with column indices of non-0 columns'''
        return np.where(X.any(axis=0))[0]

    # Visualization
    def get_graph(self):
        return nx.from_numpy_matrix(self.adjacency_matrix, create_using=nx.DiGraph)

    def draw_graph(self):
        nx.draw(self.get_graph(), with_labels=True)
