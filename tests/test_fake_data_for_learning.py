"""Tests for `fake_data_for_learning` package."""

import pytest
import numpy as np
from itertools import product

from fake_data_for_learning import GraphProbabilityTable as GPT


class TestGPTX1cX0:
    r'''
    Test the embedded probability table of X = (X0, X1)
    where
    * X0, X1 are binary
    * X admits the graph X0 -> X1
    * X0 has (marinalized) probability table (0.1, 0.9)^t
    * X1 | X0 has contitional probability table
        (0.2, 0.7)
        (0.8, 0.3)
    '''
    r = (2,2)
    #X0
    pt_X0 = np.array([0.1, 0.9])
    gpt0 = GPT(pt_X0, [0], [], [1], r)

    # X1 | X0
    pt_X1cX0 = np.array([
        [0.2, 0.7],
        [0.8, 0.3]
    ])
    gpt1c0 = GPT(pt_X1cX0, [1], [0], [], r)


    def test_embed_X0(self):
        # X0
        expected_embed_0 = np.array([
            [0.1, 0.1],
            [0.9, 0.9]
        ])
        np.testing.assert_array_equal(self.gpt0.embed_pt(), expected_embed_0)

 
    def test_embed_X1cX0(self):
        # X1 | X0
        expected_embed_1c0 = np.array([
            [0.2, 0.8],
            [0.7, 0.3]
        ])
        np.testing.assert_array_equal(self.gpt1c0.embed_pt(), expected_embed_1c0)


class TestGPTX2cX1cX0:
    r = (2,2,2)
    # X0
    pt_X0 = np.array([0.1, 0.9])
    gpt0 = GPT(pt_X0, [0], [], [1,2], r)

    # X1 | X0
    pt_X1cX0 = np.array([
        [0.2, 0.7], 
        [0.8, 0.3]
    ])
    gpt1c0 = GPT(pt_X1cX0, [1], [0], [2], r)

    # X2 | X1
    pt_X2cX1 = np.array([
        [0.8, 0.5],
        [0.2, 0.5]
    ])
    gpt2c1 = GPT(pt_X2cX1, [2], [1], [0], r)


    def test_embed_X0(self):
        # X0
        embed_0 = self.gpt0.embed_pt()
        for idx in range(2):
            assert embed_0[idx, 0, 0] == self.pt_X0[idx]
            assert embed_0[idx, 1, 0] == self.pt_X0[idx]
            assert embed_0[idx, 0, 1] == self.pt_X0[idx]
            assert embed_0[idx, 1, 1] == self.pt_X0[idx]


    def test_embed_X1cX0(self):
        # X1 | X0
        embed_1c0 = self.gpt1c0.embed_pt()
        for i,j in product(*[range(2), range(2)]):
            assert embed_1c0[i,j,0] == self.pt_X1cX0[j, i]
            assert embed_1c0[i,j,1] == self.pt_X1cX0[j, i]

    def test_embed_X2cX1(self):
        # X2 | X1
        embed_2c1 = self.gpt2c1.embed_pt()
        for i,j in product(*[range(2), range(2)]):
            assert embed_2c1[0, i, j] == self.pt_X2cX1[j, i]
            assert embed_2c1[1, i, j] == self.pt_X2cX1[j, i]
