'''Utility functions for fake_data_for_learning'''
import numpy as np
import string

trick_external_value_separator = '_'

def name_in_list(name, l):
    '''Return 1 if name is in list l, else 0'''
    if l is None:
            return 0
    res = name in l
    return res


def get_trick_external_value(val, values):
        idx = values.index(val)
        return (
            string.ascii_letters[idx] + 
            trick_external_value_separator + 
            val
        )

def untrick_external_value(val):
    return val.split('_')[-1]


def get_internal_value(external_value):
    '''
    Translate external value representation to internal one

    Parameters
    ----------
    external_value: int or dict of form {'value': int, 'le': fitted label encoder}
        External value to be translated into internal representation

    Returns
    -------
    res : int
        Internal (integer) representation of external value

    '''

    if isinstance(external_value, np.int):
        return external_value
    else:
        value = external_value.get('value')
        le = external_value.get('le')
        untricked_values = [untrick_external_value(tv) for tv in le.classes_]
        return untricked_values.index(value)



def zero_column_idx(X):
    '''Return array with column indices of 0 columns'''
    return np.where(~X.any(axis=0))[0]


def get_parent_idx(child_idx, adjacency_matrix):
    '''Return list of index positions of parents of node at child_idx in adjacency matrix'''
    res = np.nonzero(adjacency_matrix[:, child_idx])[0]
    return res.tolist()



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
    descendents = non_zero_column_idx(adjacency_matrix[parent_idx, :])

    # Keep only descendents having only parent_idx as parents
    pure_descendents = []
    for idx in descendents:
        if not adjacency_matrix[~parent_mask, idx].any():
            pure_descendents.append(idx)
    
    return np.array(pure_descendents)


def non_zero_column_idx(X):
    '''Return array with column indices of non-0 columns'''
    return np.where(X.any(axis=0))[0]
