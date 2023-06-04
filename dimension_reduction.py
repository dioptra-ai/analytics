import os
import pickle
from functools import lru_cache
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import normalize_vector

try:
    from cuml import UMAP
    cuml_loaded = True
except Exception as e:
    print(f'WARNING: CUML not loaded: {str(e)}, falling back to umap.')
    from umap import UMAP
    cuml_loaded = False

from enum import Enum
from memoize import memoize

class DimensionReductionAlgorithms(Enum):
    PCA = 'PCA'
    TSNE = 'TSNE'
    UMAP = 'UMAP'

@memoize
def reduce_dimension(dataset, algorithm, metric='euclidean', num_dimensions=2):
    if algorithm == DimensionReductionAlgorithms.PCA:
        dr_model = PCA(n_components=num_dimensions, random_state=1234)
    elif algorithm == DimensionReductionAlgorithms.TSNE:
        dr_model = TSNE(n_components=num_dimensions, random_state=1234)
    elif algorithm == DimensionReductionAlgorithms.UMAP:
        parameters = {
            'n_components': num_dimensions,
            'random_state': 1234,
            'min_dist': 0.0
        }

        if cuml_loaded:
            # CUML UMAP doesn't support cosine distance: https://docs.rapids.ai/api/cuml/nightly/api.html#umap
            # https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#gpu-acceleration
            if metric == 'cosine':
                dataset = normalize_vector(dataset)
        else:
            parameters['metric'] = metric

        dr_model = UMAP(**parameters)
    else:
        raise RuntimeError(f'No algorithm of type {algorithm}')

    dataset = np.array(dataset)
    # try:
    if len(dataset) == 1:
        return np.array([[1, 1]])

    return dr_model.fit_transform(dataset)
    # except ValueError as e:
        # raise NotEnoughDataError(1)

    # The following error is sometimes also raised here:
    # TypeError: only size-1 arrays can be converted to Python scalars
    # The above exception was the direct cause of the following exception:
    # ValueError: setting an array element with a sequence.