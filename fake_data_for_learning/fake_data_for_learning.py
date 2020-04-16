"""Main module."""
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.dag import topological_sort

from sklearn.preprocessing import LabelEncoder


class BayesianNodeRV:
    """
    Sample-able random variable corresponding to node of a discrete Bayesian
    network.

    Parameters
    ----------
    name : string
        Node variable name
    cpt: numpy.array
        (Conditional) probability array. NB: We depart from normal convention
        on the assignment of the array dimensions. In this class, the node
        variable (aka dependent, not conditioned on) corresponds to the LAST
        componend of array indexing. E.g. For a 2-d array cpt, this means that
        the ROWS must sum to 1, not the columns, as is otherwise standard. This
        choice is to make cpt definition in numpy more human-readable.

    values: list, optional
        list of values random variable will take. Default is [0, cpt.shape[-1])
        if values are given, use sklearn.preprocessing.LabelEncoder().
        NOTE: the sklearn label encoder sorts values in lexicographic order. To
        ensure compatibility with the conditional probability table, given
        values must also be in lexicographic order.

    parent_names: list, optional
        list of parent node random variable names. Default is empty list, i.e.
        no parents
    """
    def __init__(self, name, cpt, values=None, parent_names=[]):
        self.name = name
        self.cpt = cpt
        self._set_parent_names(parent_names)
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
        """Set parent_names provided no value errors raised"""
        if not isinstance(parent_names, list):
            raise TypeError('parent_names must be a list')

        if len(parent_names) + 1 != len(self.cpt.shape):
            raise ValueError(
                f'Number of parent names ({len(parent_names)}) '
                f'and conditional probability table shape ({self.cpt.shape}) '
                'are incompatible'
            )

        # Set parent names if no errors raised
        self.parent_names = parent_names

    def _set_values(self, cpt, values):
        """
        Set random variable values according to the shape of the
        conditional probability table cpt and given values.
        """
        if values is None:
            self.values = np.array(range(cpt.shape[-1]))
            self.label_encoder = None
        else:
            # Confirm that non-default values are in lexicographic order
            if not np.array_equal(
                np.array(values),
                np.unique(values)
            ):
                raise ValueError(
                    'Values must be unique and in lexicographic order'
                )
            self.label_encoder = self._set_label_encoder(values)
            self.values = self.label_encoder.classes_

    def _set_label_encoder(self, values):
        le = LabelEncoder()
        le.fit(values)
        return le

    def pmf(self, value, parent_values={}):
        """
        Probability mass function

        Parameters
        ----------
        value : int or str
            Must be contained in the class attribute `values`

        parent_values : dict of SampleValue
            Required in case of conditional random variable

        Returns
        -------
        float
        """
        check = self._validate_pmf_args(value, parent_values)
        if not check:
            raise ValueError(
                f'Input value {value} with parent values {parent_values} '
                f'incompatible with allowed values {self.values} and '
                f'parent names {self.parent_names}'
            )

        res = self._get_pmf(value, parent_values)
        return res

    def _validate_pmf_args(self, value, parent_values):
        checks = []

        checks.append(value in list(self.values))

        if parent_values == {}:
            checks.append(self.parent_names == [])
        elif self.parent_names == []:
            checks.append(parent_values == {})
        else:
            checks.append(
                set(parent_values.keys()).issubset(set(self.parent_names))
            )

        res = np.array(checks).all()
        return res

    def _get_pmf(self, value, parent_values):
        if self.label_encoder is None:
            idx = value
        else:
            idx = self.label_encoder.transform([value])[0]

        probability_table = self.get_probability_table(parent_values)
        return probability_table[idx]

    def rvs(self, parent_values={}, size=1, seed=None):
        """
        Generate random variates from the bayesian node.

        Parameters
        -----------
        parent_values : dict
            Dict with parent values, possibly empty
        size : int
            Number of random samples to draw
        seed : int
            Seed for numpy.random
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.
        """
        np.random.seed(seed)

        if self.parent_names == []:
            return np.random.choice(self.values, size, p=self.cpt)
        else:
            res = np.random.choice(
                self.values, size,
                p=self.get_probability_table(parent_values)
            )
            return res

    def get_probability_table(self, parent_values={}):
        """
        Get probability table.

        Parameters
        ----------
        parent_values: dict, empty or {'name1': SampleValue(...), 'name2': ...}
            Values of parent nodes to get relevant 1-d submatrix of the
            (conditional) probability table
        """
        if parent_values == {}:
            return self.cpt
        else:
            s = [slice(None)] * len(self.cpt.shape)
            for idx, p in enumerate(self.parent_names):
                parent_internal_value = self.get_internal_value(
                    parent_values[p]
                )
                s[idx] = parent_internal_value
            return self.cpt[tuple(s)]

    def get_internal_value(self, sample_value):
        """
        Translate SampleValue representation to natural number consistent with
        conditional probability table definition.

        Parameters
        ----------
        sample_value: SampleValue

        Returns
        -------
        res : int
            Internal (integer) representation of external value

        """
        if sample_value.label_encoder is not None:
            return sample_value.label_encoder.transform(
                [sample_value.value]
            )[0]
        else:
            return sample_value.value

    def __str__(self):
        return 'BayesianNodeRV({}, parent_names={})'.format(
            self.name, self.parent_names
        )


class SampleValue:
    """
    Container for parent values when sampling from conditional probability
    distributions.

    If the sample values are non-default--i.e. other than natural numbers
    corresponding to the number of possible variable states--then a label
    encoder is required.
    """

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
            if value not in self.label_encoder.classes_:
                raise ValueError('Value has not been encoded')
            else:
                return value

    @staticmethod
    def possible_default_value(x):
        """
        Check conditions that rule-out a default (i.e. natural number) value.
        """
        if isinstance(x, np.int) or isinstance(x, np.int64):
            return x >= 0
        else:
            return False

    def __str_(self):
        return 'SampleValue({}, {})'.format(self.value, self.label_encoder)


class FakeDataBayesianNetwork:
    """
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
        Adjacency matrix of the Bayesian network's graph
    eve_node_names: list of strings
        Node variable names without parents
    """
    def __init__(self, *args):
        self.bnrvs = args
        self.node_names = self._set_node_names()
        self.adjacency_matrix = self.calc_adjacency_matrix()
        self._validate_bn()
        self.graph = self.get_graph()
        self.topoligically_ordered_node_names = self.get_topological_ordering()

    def _set_node_names(self):
        node_names = []
        parent_names = []

        for rv in self.bnrvs:
            node_names.append(rv.name)
            if rv.parent_names is not None:
                parent_names += rv.parent_names

        missing_nodes = set(parent_names) - set(node_names)

        if missing_nodes != set():
            raise ValueError('Missing nodes from network: {}'.format(
                missing_nodes
            ))

        return node_names

    def _validate_bn(self):
        """
        Check for consistency of node random variables in Bayesian network
        """

        # Check consistency of conditional probability tables between parents
        # and children
        for idx, rv in enumerate(self.bnrvs):
            parent_idxs = self.get_parent_idx(idx, self.adjacency_matrix)
            expected_cpt_dims = self.get_expected_cpt_dims(
                parent_idxs, len(rv.values)
            )

            if rv.cpt.shape != tuple(expected_cpt_dims):
                raise ValueError(
                    '{} conditional probability table dimensions {} '
                    'inconsistent with parent values {}'.format(
                        rv.name, rv.cpt.shape, expected_cpt_dims
                    )
                )

    @staticmethod
    def get_parent_idx(child_idx, adjacency_matrix):
        """
        Return list of index positions of parents of node at child_idx in
        adjacency matrix
        """
        res = np.nonzero(adjacency_matrix[:, child_idx])[0]
        return res.tolist()

    def get_expected_cpt_dims(self, parent_idxs, child_value_length):
        """
        Return expected dimension of the conditional probability table.
        Note that the tuple ordering depends on the ordering of the
        parent ordering for each non-orphan bayesian node random variable.

        Parameters
        ----------
        parent_idxs : list of int
            Indices of parent nodes in creation of Bayesian network
        child_value_length : int
            Cardinality of values of child node

        Returns
        -------
        tuple
        """
        expected_cpt_dims = []
        for parent_idx in parent_idxs:
            expected_cpt_dims.append(len(self.bnrvs[parent_idx].values))

        # append node value length
        expected_cpt_dims.append(child_value_length)
        return tuple(expected_cpt_dims)

    def calc_adjacency_matrix(self):
        res = np.zeros((len(self.bnrvs), len(self.bnrvs)), dtype=int)
        for i, node_i in enumerate(self.bnrvs):
            for j, node_j in enumerate(self.bnrvs):
                res[i, j] = FakeDataBayesianNetwork.name_in_list(
                    node_i.name, node_j.parent_names
                )
        return res

    @staticmethod
    def name_in_list(name, l):
        """Return 1 if name is in list l, else 0."""
        if l is None:
            return 0
        res = name in l
        return res

    def get_graph(self):
        g = nx.from_numpy_matrix(
            self.adjacency_matrix, create_using=nx.DiGraph
        )
        # Add node labels
        labels = {n: self.node_names[n] for n in range(len(self.node_names))}
        g = nx.relabel_nodes(g, labels)
        return g

    def get_topological_ordering(self):
        return list(topological_sort(self.graph))

    def rvs(self, size=1):
        """Ancestrally sampled values from the Bayesian network."""
        res = [self.get_ancestral_sample() for _ in range(size)]
        return pd.DataFrame.from_records(
            res, index=range(size), columns=self.node_names
        )

    def get_ancestral_sample(self):
        """Ancestral sampling from the Bayesian network."""
        res = {}
        res = self.get_ordered_samples()
        # Keep only SampleValue value
        res = {k: v.value for (k, v) in res.items()}
        return res

    def get_ordered_samples(self):
        res = {}
        for node_name in self.topoligically_ordered_node_names:
            node = self.get_node(node_name)
            res[node_name] = SampleValue(
                node.rvs(
                    size=1,
                    parent_values=res,
                )[0],
                label_encoder=node.label_encoder
            )
        return res

    def get_node(self, node_name):
        if node_name not in self.node_names:
            raise ValueError('No node defined with name {}'.format(node_name))
        res = self.bnrvs[self.node_names.index(node_name)]
        return res

    def pmf(self, sample):
        """
        Probabilty mass function for bayesian network.

        Parameters
        ----------
        sample : pandas.Series
            Series of sampled values

        Returns
        -------
        res : float
            Probability mass function evaluated at sample
        """
        res = 1.
        for i in range(len(sample)):
            res *= self._get_ith_pmf(sample, i)

        return res

    def _get_ith_pmf(self, sample, i):
        child = self.bnrvs[i]
        parent_idx = self.get_parent_idx(i, self.adjacency_matrix)

        if not parent_idx:
            parent_values = {}
        else:
            parent_values = {
                self.bnrvs[idx].name: SampleValue(
                    sample[idx],
                    label_encoder=self.bnrvs[idx].label_encoder
                )
                for idx in parent_idx
            }

        return child.pmf(sample[i], parent_values=parent_values)

    # Visualization
    def draw_graph(self):
        nx.draw_networkx(self.get_graph(), node_size=800, node_color='#00b4d9')

    def __str__(self):
        return 'FakeDataBayesianNetwork with node_names={})'.format(
            self.node_names
        )
