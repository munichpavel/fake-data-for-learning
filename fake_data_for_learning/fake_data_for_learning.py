# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder

from . import utils as ut

from .utils import trick_external_value_separator


class BayesianNodeRV:
    '''
    Sample-able random variable corresponding to node of a discrete Bayesian network.

    Parameters
    ----------
    name : string
        Node variable name
    cpt: numpy array
        (Conditional) probability array. NB: We depart from normal convention on the assignment of the 
        array dimensions. In this class, the node variable (aka dependent, not conditioned on) corresponds
        to the LAST componend of array indexing. E.g. For a 2-d array cpt, this means that the ROWS must sum to 1,
        not the columns, as is otherwise standard. This choice is to make cpt definition in numpy more human-readable.

    values: list, optional
        list of values random variable will take. Default is [0, cpt.shape[-1])

    parents: list, optional
        list of parent node random variable names. Default is None, i.e. no parents
    '''
    def __init__(self, name, cpt, values=None, parent_names=None):
        self.name = name
        self.cpt = cpt
        self.parent_names = parent_names
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

    def _set_values(self, cpt, values):
        if values is None:
            self.values = np.array(range(cpt.shape[0]))
            self.le = None
            self._values = self.values
        else:
            self._set_nondefault_values(values)

    def _set_nondefault_values(self, values):
        # external values representation
        self._validate_nondefault_values(values)
        self.values = values

        # label encoder
        le = LabelEncoder()
        # Hack to trick out sklearn's baked in sort order
        tricked_external_values = []
        for val in values:
            tricked_external_values.append(ut.get_trick_external_value(val, values))
        le.fit(tricked_external_values)
        self.le = le

        # internal values representation
        self._values = le.transform(tricked_external_values)

    def _validate_nondefault_values(self, values):
        for val in values:
            print(trick_external_value_separator)
            if trick_external_value_separator in val:
                raise ValueError(
                    'The character {} is not permitted in values'
                    .format(trick_external_value_separator)
                ) 
    
    def rvs(self, parent_values=None, size=None, seed=None):
        '''
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
        '''
        Parameters
        ----------
        parent_values: None, dict of ints of form {'node_name': int}, 
                or dict of dicts with entries 'node_name': {'value': int, 'le': fitted label encoder}
            Values of parent nodes to get relevant 1-d submatrix of the (conditional) probability table
        '''
        if parent_values is None:
            return self.cpt
        else:
            s = [slice(None)] * len(self.cpt.shape)
            for idx, p in enumerate(self.parent_names):
                parent_internal_value = ut.get_internal_value(parent_values[p])
                s[idx] = parent_internal_value
                #s[idx] = parent_values[p]
            return self.cpt[tuple(s)]



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
            parent_idxs = ut.get_parent_idx(idx, self.adjacency_matrix)
            expected_cpt_dims = self._get_expected_cpt_dims(parent_idxs, len(rv.values))
            if rv.cpt.shape != tuple(expected_cpt_dims):
                raise ValueError(
                    'Conditional probability table dimensions {} of {} inconsistent with parent values {}'.format(
                    rv.cpt.shape, rv.name, expected_cpt_dims)
                )

    def _get_expected_cpt_dims(self, parent_idxs, child_value_length):
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
                res[i,j] = ut.name_in_list(node_i.name, node_j.parent_names)
        return res

    def _set_eve_node_names(self):
        r'''Find eve nodes as zero columns of adjacency matrix'''
        eve_idx = ut.zero_column_idx(self.adjacency_matrix)
        res = list(np.array(self.node_names)[eve_idx])
        return res

    def rvs(self, seed=None):
        r'''Ancestral sampling from the Bayesian network'''
        samples_dict = {}
        sample_next_names = self._eve_node_names
        while set(samples_dict.keys()) != set(self.node_names):
            for node_name in sample_next_names:
                node = self._bnrvs[self.node_names.index(node_name)]
                print(samples_dict)
                sample = node.rvs(samples_dict, seed=seed)
                if isinstance(sample, np.int) or isinstance(sample, np.int64):
                    samples_dict[node_name] = sample
                else:
                    samples_dict[node_name] = {'value': sample, 'le': node.le}
     
            idx_current_names = np.array([self.node_names.index(name) for name in sample_next_names])
            idx_next_names = ut.get_pure_descendent_idx(idx_current_names, self.adjacency_matrix)
            sample_next_names = [self.node_names[idx] for idx in idx_next_names]

        res = ut.flatten_samples_dict(samples_dict)
        return pd.DataFrame(res, index=range(1), columns=self.node_names)

        # samples_dict = {}
        # sample_next_names = self._eve_node_names
        # while set(samples_dict.keys()) != set(self.node_names):
        #     for node_name in sample_next_names:
        #         node = self._bnrvs[self.node_names.index(node_name)]
        #         samples_dict[node_name] = node.rvs(samples_dict, seed=seed)
            
        #     idx_next_names = np.array([self.node_names.index(name) for name in sample_next_names])
        #     sample_next_names = ut.get_pure_descendent_idx(idx_next_names, self.adjacency_matrix)

        # return samples_dict

    


    # def rvs(self, seed=None):
    #     '''
    #     Ancestral sampling from Bayesian network.
    #     '''
    #     res = np.array(len(self.node_names) * [np.nan])
    #     # First get eve node component indices
    #     idx_sample_next = np.array([self.node_names.index(eve) for eve in self._eve_node_names])

    #     samples_dict = {}
    #     while np.isnan(res).any():
    #         # Sample next round of nodes given values in sample_dict
    #         for idx in idx_sample_next:
    #             node = self._bnrvs[idx]
    #             res[idx] = node.rvs(samples_dict, seed=seed)
    #         samples_dict = self._sample_array_to_dict(res) 

    #         idx_sample_next = ut.get_pure_descendent_idx(idx_sample_next, self.adjacency_matrix)
     
    #     return res

    def _sample_array_to_dict(self, res_array):
        r'''
        Convert sampled result array of form (x0, ..., xn)
        to dict of form {'X0': x0, ..., 'Xn': xn}.
        '''

        samples_dict = {}
        idx_sampled = np.where(~np.isnan(res_array))[0]
        for idx in idx_sampled:
            #sample_dict[self.node_names[idx]] = int(res_array[idx])
            samples_dict[self.node_names[idx]] = self._get_sample_dict(idx, res_array)

        return samples_dict

    def _get_sample_dict(self, idx, res_array):
        sample = res_array[idx]
        if isinstance(sample, np.float):
            res = int(sample)
        else:
            res = {
                'le': self._bnrvs[idx].le,
                'value': sample
            }
        print(res)
        return res

    def get_graph(self):
        return nx.from_numpy_matrix(self.adjacency_matrix, create_using=nx.DiGraph)

    def draw_graph(self):
        nx.draw(self.get_graph(), with_labels=True)
