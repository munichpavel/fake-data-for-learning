# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np


class BayesianNodeRV:
    '''
    Sample-able random variable corresponding to node of a discrete Bayesian network.
    '''
    def __init__(self, name, cpt, values=None, parents=None):
        
        self.name = name
        self.cpt = cpt
        self.parents = parents
        self.values = self._set_values(cpt, values)


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

        if self.parents is None:
            return np.random.choice(self.values, size, p=self.cpt)
        else:
            res = np.random.choice(self.values, size, p=self.get_pt(parent_values))
            return res
    

    def get_pt(self, parent_values=None):
        if parent_values is None:
            return self.cpt
        else:
            s = [slice(None)] * len(self.cpt.shape)
            for idx, p in enumerate(self.parents):
                s[idx + 1] = parent_values[p]
            return self.cpt[tuple(s)]


class FakeDataBayesianNetwork:
    '''
    Sample-able Bayesian network comprised up of BayesianNetworkRV's
    '''
    def __init__(self, *args):
        self._bnrvs = args
        self.node_names = self._set_node_names()
        self.adjacency_matrix = self.calc_adjacency_matrix()

    
    def _set_node_names(self):

        node_names = []
        parent_names = []

        for rv in self._bnrvs:
            node_names.append(rv.name)
            if rv.parents is not None:
                parent_names += rv.parents

        missing_nodes = set(parent_names) - set(node_names)

        if missing_nodes != set():
            raise ValueError('Missing nodes from network: {}'.format(missing_nodes))
        
        return node_names

    def calc_adjacency_matrix(self):
        
        res = np.zeros((len(self._bnrvs), len(self._bnrvs)), dtype=int)
        for i, node_i in enumerate(self._bnrvs):
            for j, node_j in enumerate(self._bnrvs):
                res[i,j] = self._name_in_list(node_i.name, node_j.parents)
        return res


    @staticmethod
    def _name_in_list(name, l):
        '''Return 1 if name is in list l, else 0'''
        if l is None:
             return 0
        
        res = name in l
        return res



