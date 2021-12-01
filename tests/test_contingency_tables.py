from pandas.api.types import CategoricalDtype
import pandas as pd
import numpy as np
import xarray as xr

import pytest

from fake_data_for_learning.contingency_tables import (
    get_shape_from_coords,
    calculate_contingency_table,
    generate_fake_data_for_contingency_table
)


@pytest.mark.parametrize(
    'contingency_values,data_categories,ExpectedException',
    [
        (
            [[0, 1], [0, 0]],
            dict(
                recovered=CategoricalDtype(categories=[0, 1], ordered=True),
                treated=CategoricalDtype(categories=[0, 1], ordered=True)
            ),
            None
        ),
        (
            [[0, 0], [0, 0]],
            dict(
                recovered=CategoricalDtype(categories=[0, 1], ordered=True),
                treated=CategoricalDtype(categories=[0, 1], ordered=True)
            ),
            None
        ),
        (
            [[10, 5], [2, 42]],
            dict(
                recovered=CategoricalDtype(categories=[0, 1], ordered=True),
                treated=CategoricalDtype(categories=[0, 1], ordered=True)
            ),
            None
        ),
        (
            [
                [1, 1, 1],
                [1, 1, 1]],
            dict(
                recovered=CategoricalDtype(categories=[0, 1], ordered=True),
                treated=CategoricalDtype(categories=[0, 1, 2], ordered=True)
            ),
            None
        ),
        (
            [[
                [1, 0],
                [0, 0]
            ], [
                [1, 2],
                [5, 2]
            ]],
            dict(
                recovered=CategoricalDtype(categories=[0, 1], ordered=True),
                gender=CategoricalDtype(categories=[0, 1], ordered=True),
                treated=CategoricalDtype(categories=[0, 1], ordered=True)
            ),
            None
        ),
        (
            [[0, 1], [0, 0]],
            dict(
                recovered=CategoricalDtype(categories=['no', 'yes'], ordered=True),
                treated=CategoricalDtype(categories=[0, 1], ordered=True)
            ),
            NotImplementedError
        ),
    ]
)
def test_generate_fake_data_for_contingency_table(
    contingency_values, data_categories, ExpectedException
):
    coords = {}
    for column, c_type in data_categories.items():
        coords[column] = c_type.categories
    contingency_table = xr.DataArray(
        contingency_values,
        dims=data_categories.keys(),
        coords=coords
    )

    if ExpectedException is None:
        data = generate_fake_data_for_contingency_table(contingency_table)
        # Convert to categorical dtype for computing sampled data contingency table
        for column, c_type in data_categories.items():
            data[column] = data[column].astype(c_type)

        xr.testing.assert_equal(
            calculate_contingency_table(data),
            contingency_table
        )
    else:
        with pytest.raises(ExpectedException):
            generate_fake_data_for_contingency_table(contingency_table)



@pytest.mark.parametrize(
    'data_values,data_categories,expected,ExpectedException',
    [
        (
            [[1, 0], [0, 1]],
            dict(
                recovered=CategoricalDtype(categories=[0, 1], ordered=True),
                treated=CategoricalDtype(categories=[0, 1], ordered=True)
            ),
            xr.DataArray(
                [
                    [0, 1],
                    [1, 0]
                ],
                dims=('recovered', 'treated'),
                coords=[[0, 1], [0, 1]]
            ),
            None
        ),
        (
            [[1, 0], [0, 1]],
            dict(recovered=int, treated=int),
            xr.DataArray(
                [
                    [0, 1],
                    [1, 0]
                ],
                dims=('recovered', 'treated'),
                coords=[[0, 1], [0, 1]]
            ),
            AttributeError
        ),
    ]
)
def test_calculate_contingency_table(
    data_values, data_categories, expected, ExpectedException
):

    data = pd.DataFrame(data_values, columns=data_categories.keys())
    for column, c_type in data_categories.items():
        data[column] = data[column].astype(c_type)
    if ExpectedException is None:
        contingency_table = calculate_contingency_table(data)

        xr.testing.assert_equal(contingency_table, expected)
    else:
        with pytest.raises(ExpectedException):
            calculate_contingency_table(data)


@pytest.mark.parametrize(
    'coords,expected,ExpectedException',
    [
        (dict(recovered=[0, 1], treated=[0, 1]), [2, 2], None),
        (dict(recovered=(0, 1), treated=(0, 1)), [2, 2], None),
        (dict(recovered=pd.Series([0, 1]), treated=np.array([0, 1])), [2, 2], None),
        (
            dict(recovered=[0, 1], treated=[0, 1], whacked=[0, 1, 2]),
            [2, 2, 3], None
        ),
        (dict(), [], None),
        (dict(flat=[0, 1], not_flat=[[0, 1], 1]), [2, 2], None),
        (dict(chutzpah='bagel'), None, TypeError)
    ]
)
def test_get_shape_from_coords(coords, expected, ExpectedException):
    if ExpectedException is None:
        res = get_shape_from_coords(coords)
        assert res == expected
    else:
        with pytest.raises(ExpectedException):
            get_shape_from_coords(coords)


