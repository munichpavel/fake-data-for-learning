# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np
from . import utils as ut


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
        self.values = self._set_values(cpt, values)

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
            return np.array(range(cpt.shape[0]))
        else:
            return values
 
    def rvs(self, parent_values=None, size=None, seed=42):
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
        if parent_values is None:
            return self.cpt
        else:
            s = [slice(None)] * len(self.cpt.shape)
            for idx, p in enumerate(self.parent_names):
                s[idx] = parent_values[p]
            return self.cpt[tuple(s)]


class FakeDataBayesianNetwork:
    '''
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
 
    def calc_adjacency_matrix(self):
        
        res = np.zeros((len(self._bnrvs), len(self._bnrvs)), dtype=int)
        for i, node_i in enumerate(self._bnrvs):
            for j, node_j in enumerate(self._bnrvs):
                res[i,j] = ut.name_in_list(node_i.name, node_j.parent_names)
        return res

    def _set_eve_node_names(self):
        '''Find eve nodes as zero columns of adjacency matrix'''
        eve_idx = np.where(~self.adjacency_matrix.any(axis=0))[0]
        res = list(np.array(self.node_names)[eve_idx])
        return res

    def _sample_eves(self, seed=42):
        res = np.array(len(self.node_names) * [np.nan])
        
        for eve in self._eve_node_names:
            idx = self.node_names.index(eve)
            node = self._bnrvs[idx]
            res[idx] = node.rvs(seed)

        return res
