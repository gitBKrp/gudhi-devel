""" This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
    Author(s):       Bazyli Kot

    Copyright (C) 2019 Inria

    Modification(s):
      - YYYY/MM Author: Description of the modification
"""


import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from gudhi.representations.vector_methods import PersistenceBow, StablePersistenceBow
from gudhi.representations.preprocessing import RandomPDSampler, GridPDSampler
__author__ = "Bazyli Kot"
__copyright__ = "Copyright (C) 2019 Inria"
__license__ = "MIT"

def test_empty_input():
    pbow = PersistenceBow(KMeans())
    assert isinstance(pbow.fit_transform([]), np.ndarray)
    assert pbow.fit_transform([]).size == 0
    spbow = estimator = StablePersistenceBow(GaussianMixture())
    assert isinstance(spbow.fit_transform([]), np.ndarray)
    assert spbow.fit_transform([]).size == 0

def test_zero_input_ft():
    pbow = PersistenceBow(KMeans(n_clusters=1))
    # 4 zeros in single cluster ending in vector [4] then normalized
    assert np.all(pbow.fit_transform([np.zeros((4, 2))]) == [np.array([0.5])])
    spbow  = StablePersistenceBow(GaussianMixture())
    assert np.allclose(spbow.fit_transform([np.zeros((4, 2))]), [[0.5]])

def test_pbow_standard_case():
    # Generate three distinctive clouds of 10 points
    centers = [[10, 10], [20, 20], [30, 30]]
    _input = [np.array([np.array(c) + np.random.normal()
                        for i in range(10)])
              for c in centers]
    pbow = PersistenceBow(KMeans(n_clusters=3, 
                                init=np.array(centers),
                                random_state=1),
                                scaler=None,
                                transformator=None,
                                normalize=False)

    assert np.all(pbow.fit_transform(_input) == [[10, 0, 0],
                                                 [0, 10, 0],
                                                 [0, 0, 10]])

def test_execute_all():
    _input = [np.array([[ c + np.random.normal(), c + np.random.normal()]
                        for i in range(10)])
              for c in range(10, 30, 10)]
    pbow = PersistenceBow(KMeans(n_clusters=1),
                          sampler=RandomPDSampler(max_points=1, 
                                                  weight_function=lambda x: 1),
                          cluster_weighting=lambda x: x[1])
    pbow.fit_transform(_input, sample_weight=1)
    
    pbow = PersistenceBow(KMeans(n_clusters=1),
                          sampler=GridPDSampler(grid_shape=(2, 2), 
                                                max_points=4,
                                                weight_function=lambda x: 1),
                          cluster_weighting=lambda x: 1)
    pbow.fit_transform(_input, sample_weight=1)
    
    spbow = StablePersistenceBow(GaussianMixture(),
                          sampler=RandomPDSampler(max_points=2, 
                                                  weight_function=lambda x: 1),
                          cluster_weighting=lambda x: x[1])
    
    spbow.fit_transform(_input)
    
    spbow = StablePersistenceBow(GaussianMixture(),
                                 sampler=GridPDSampler(grid_shape=(2, 2), 
                                 max_points=2,
                                 weight_function=lambda x: 1),
                                 cluster_weighting=lambda x: 1)
    
    spbow.fit_transform(_input)    
