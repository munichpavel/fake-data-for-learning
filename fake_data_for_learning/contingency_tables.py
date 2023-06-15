"""Methods for generative fake categorical data to match contingency tables"""

import pandas as pd
import numpy as np
import xarray as xr

from itertools import product


def generate_fake_data_for_contingency_table(
    contingency_table: xr.DataArray, seed=42
) -> pd.DataFrame:
    """Generate fake data whose contingency table matches the given one"""
    dims = contingency_table.dims
    idxes = []
    for dim in dims:
        idxes.append(contingency_table.get_index(dim))

    events = list(product(*idxes))
    flat_contingency_table = pd.Series(np.zeros(len(events)), index=events, dtype=int)
    for event in events:
        try:
            flat_contingency_table[event] = contingency_table[event]
        except TypeError:
            msg = f'Only integer coordinate values supported, entered coord was {event}'
            raise NotImplementedError(msg)

    samples = []
    for idx, count in flat_contingency_table.items():
        samples += count * [idx]
    res = pd.DataFrame(samples, columns=dims)
    # Shuffle rows uniformly so it looks better
    res = res.sample(
        res.shape[0], replace=False, random_state=seed
    ).reset_index(drop=True)
    return res


def calculate_contingency_table(data: pd.DataFrame) -> xr.DataArray:
    """Calculate a contingency table given categorical data"""
    counts = data.groupby(data.columns.tolist(), as_index=True).size()

    dims = data.columns
    coords = dict()
    for dim in dims:
        c_type = data.dtypes[dim]
        coords[dim] = c_type.categories.values

    # Calculate shape of xarray from categories
    shape = get_shape_from_coords(coords)

    res = xr.DataArray(np.zeros(shape), dims=dims, coords=coords)

    for idx in counts.index:
        res[idx] = counts[idx]

    return res


def get_shape_from_coords(coords: dict):
    """Get the shape of a multi-dimensional array from an xarray coords-like dict"""
    _iterable_types = [np.ndarray, list, tuple, pd.Series]
    res = []
    error_msg = ''
    for values in coords.values():
        if type(values) not in _iterable_types:
            error_msg += (
                f'Coordinate value {values} is not among expected types: '
                f'{_iterable_types}'
            )
        res.append(len(values))

    if error_msg != '':
        raise TypeError(error_msg)
    return res
