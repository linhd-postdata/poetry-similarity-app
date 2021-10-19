# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""

from itertools import combinations
from typing import List

import numpy as np
import requests
from elasticsearch import Elasticsearch
from flask import flash

from .settings import ES_HOST
from .settings import ES_INDEX
from .settings import ES_PORT
from .models import MODELS

ES = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT}])
OPERATIONS = {"sum",
              "avg_lr", "avg_rl",
              "ind_lr", "ind_rl",
              "jnt_lr", "jnt_rl",
              "inf_rl", "inf_lr"}


def flash_errors(form, category="warning"):
    """Flash all errors for a form."""
    for field, errors in form.errors.items():
        for error in errors:
            flash(f"{getattr(form, field).label.text} - {error}", category)


def query_es(q, index="poems_metadata"):
    """Get the ElasticSearch response to the query.
        Query can be on `artist`, `title` or `text`.

    :param q: query to be executed
    :type q: str
    :param index: ElasticSearch index where the query will be launched
    :type index: str
    :return: :class:`Response` object
    :rtype: Response
    """
    params = (
        ('pretty', ''),
        ('q', q),
    )
    response = requests.get(f"http://{ES_HOST}:{ES_PORT}/{index}/_search", params=params)
    if response.status_code == 200:
        response = response.json()
    else:
        response = None
    return response


def get_similar_es(poem, similarity_base):
    """Get the similar poems to the given POEM based on the SIMILARITY_BASE
        parameter.

    :param poem: selected poem to search similar ones.
    :type poem: dict
    :param similarity_base: dict with information for the ES index and
        similarity function used to calculate the poem similarity
    :type similarity_base: dict
    :return: :class:`Response` object from ElasticSearch
    :rtype: Response
    """
    q = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc[params.composition_function]) + 1.0",
                "params": {
                    "query_vector": poem[MODELS[similarity_base]["composition_fn"]],
                    "composition_function": MODELS[similarity_base]["composition_fn"]
                }
            }
        }
    }
    body = {"query": q, "size": 10}
    response = ES.search(index=ES_INDEX, body=body)
    return response


def add_poem_metadata(poem_dict):
    """Upload poem info with metadata from ElasticSearch

    :param poem_dict: query to be executed
    :type poem_dict: dict
    :return: dict with `author` and `title` added to the poem
    :rtype: dict
    """
    q = f'song_id:{poem_dict["_source"]["song_id"]}'
    poem_metadata = query_es(q, "poems_metadata")["hits"]["hits"][0]["_source"]
    poem_dict["_source"].update({"author": poem_metadata["author"],
                                 "poem_title": poem_metadata["poem_title"]})
    return poem_dict


OPERATIONS = {"sum",
              "avg_lr", "avg_rl",
              "ind_lr", "ind_rl",
              "jnt_lr", "jnt_rl",
              "inf_rl", "inf_lr"}


def generalized_composition_function(alpha: float, beta: float,
                                     v1: np.ndarray,
                                     v2: np.ndarray) -> np.ndarray:
    r"""
    The generalized composition function
    :math:`F_{α,β}∶{\mathbb R}^n × {\mathbb R}^n \rightarrow {\mathbb R}`
    is defined as:

    .. math::
        F_{α,β}(\vec{v}_1,\vec{v}_2) = \frac{\vec{v}_1+\vec{v}_2} {\|\vec{v}_1+\vec{v}_2\|}
        \sqrt {\alpha(\|\vec{v}_1\|^2+\|\vec{v}_2\|^2)-\beta\left\langle \vec{v}_1, \vec{v}_2 \right\rangle}

    This function can lead to:
        - :math:`F_{Sum}` when α=1 and β=−2
        - :math:`F_{Avg}` when α=1/4 and β=1/2
        - :math:`F_{Ind}` when α=1 and β=0
        - :math:`F_{Joint}` when α=1 and β=1
        - :math:`F_{Inf}` when α=1 and β= :math:`\frac{min(\|\vec{v}_1\|,\|\vec{v}_2\|)} {max(\|\vec{v}_1\|,\|\vec{v}_2\|)}`

    :param alpha: parameter of the generalized composition function
    :type alpha: float
    :param beta: parameter of the generalized composition function
    :type beta: float
    :param v1: composition vector 1
    :type v1: np.ndarray
    :param v2: composition vector 2
    :type v2: np.ndarray
    :return: Numpy array resulting of composition function
    :rtype: np.ndarray
    """
    vector_sum = v1 + v2
    norm_vector_sum = np.linalg.norm(vector_sum)
    alpha_part = alpha * (np.linalg.norm(v1) ** 2 + np.linalg.norm(v2) ** 2)
    beta_part = beta * np.dot(v1, v2)
    square_root = np.sqrt(abs(alpha_part - beta_part))
    composite_vector = vector_sum / norm_vector_sum * square_root
    return composite_vector


def icm(beta: float, v1: np.ndarray, v2: np.ndarray):
    r"""
    The Information Contrast Model (ICM) (Amigó et al. 2020b) is ageneralization
    of the well-known PointWise Mutual Information (PMI).
    Assumming symmetricity (α1=α2=1), and assuming that the inner product
    approaches the PMI of the projected linguistic units, we can define
    the vector-based ICM as:

    .. math::
        ICM^V_{β} = (1 - \beta)(\|\vec{v}_1\|^2 + \|\vec{v}_2\|^2) +
        \beta \left\langle \vec{v}_1, \vec{v}_2 \right\rangle

    Interestingly, :math:`ICM^V` is equivalent to the inner product when β=1,
    and it is equivalent to the euclidean distance when β=2.
    In addition, :math:`ICM^V` is the only one that satisfies the three
    similarity constraints.

    :param beta: parameter of the ICM function
    :type beta: float
    :param v1: Embedding vector 1
    :type v1: np.ndarray
    :param v2: Embedding vector 1
    :type v2: np.ndarray
    :return: Similarity between v1 and v2
    """
    left_part = 1 - beta * (np.linalg.norm(v1) ** 2 + np.linalg.norm(v2) ** 2)
    right_part = beta * np.dot(v1, v2)
    icm_similarity = left_part + right_part
    return icm_similarity


def f(operation: str, v1: np.ndarray, v2: np.ndarray):
    """Calculates the chosen operation between two vectors

    :param operation: A string with the possible values for the operations
    :type operation: str
    :param v1: Embedding vector 1
    :type v1: np.ndarray
    :param v2: Embedding vector 2
    :type v2: np.ndarray
    :return: Numpy array with the result of the f function
    :rtype: np.ndarray
    """
    op_dict={"sum": (1,-2),
             "avg": (0.25, 0.5),
             "ind": (1,0),
             "jnt": (1,1),
             "inf": (1, calc_beta_finf(v1, v2))
             }
    return generalized_composition_function(*op_dict[operation], v1, v2)


def get_composite_vector(operation: str, embeddings_list: List[np.array], r2l=False):
    """Calculates the chosen operation for a list of arrays

    :param operation: A string with the possible values for the operations
    :type operation: str
    :param embeddings_list: A list of embeddings
    :type embeddings_list: List[np.array]
    :param r2l: A flag to calculate the composition vector from right to left or
        viceversa
    :type r2l: bool
    :return: Numpy array with the result of the f function
    :rtype: np.ndarray
    """
    embeddings_list = embeddings_list[::]
    if r2l:
        embeddings_list = embeddings_list[::-1]
    if embeddings_list:
        v1 = embeddings_list.pop(0)
    else:
        raise ValueError('Empty list of embeddings...')
    while embeddings_list:
        v2 = embeddings_list.pop(0)
        v1 = f(operation, v1, v2)
    return v1


def get_average(embeddings_list: List[np.array]):
    """Standard average function

    :param embeddings_list: A list of embeddings
    :type embeddings_list: List[np.array]
    :return: Numpy array with the result of the average function
    :rtype: np.ndarray
    """
    average = np.stack(embeddings_list, axis=0)
    return np.mean(average, axis=0)


def composition_function(f_name: str, embedding_list: List[np.array]):
    """Calculates the chosen operation for a list of arrays

    :param f_name: A string with the possible values for the operations
    :type f_name: str
    :param embeddings_list: A list of embeddings
    :type embeddings_list: List[np.array]
    :return: Numpy array with the result of the average function
    :rtype: np.ndarray
    """
    r2l = False
    if '_rl' in f_name:
        r2l = True
    if f_name in OPERATIONS:
        f_name = f_name[:3]
        output = get_composite_vector(f_name, embedding_list, r2l)
    elif f_name == 'avg':
        output = get_average(embedding_list)
    else:
        raise ValueError(f'{f_name} - this composition function is not implemented')

    if (output is None) or (np.isnan(output).any()) or (output.shape[0] != embedding_list[0].shape[0]):
        raise ValueError('Not correct output')

    return output


def calc_beta_finf(v1, v2):
    r"""
    Function to calculate beta value in Inf composition function:

     .. math::
        β = \frac{min(\|\vec{v}_1\|,\|\vec{v}_2\|)} {max(\|\vec{v}_1\|,\|\vec{v}_2\|)}

    :param v1: composition vector 1
    :type v1: np.ndarray
    :param v2: composition vector 2
    :type v2: np.ndarray
    :return: Result of the formula
    :rtype: float
    """
    if np.linalg.norm(v1) > np.linalg.norm(v2):
        minimum = np.linalg.norm(v2)
        maximum = np.linalg.norm(v1)
    else:
        minimum = np.linalg.norm(v1)
        maximum = np.linalg.norm(v2)
    beta_finf = minimum / maximum
    return beta_finf


def calc_beta_icm(embeddings_list: List[np.array]):
    r"""
    Function to calculate an estimation of the optimal beta parameter for a list
    of token embeddings.

    .. math::
        β = \frac {avg \left( \|\vec{v}_i\|^2+\|\vec{v}_j\|^2 \right)}
        {avg \left( \|\vec{v}_i\|^2+\|\vec{v}_j\|^2 - \left\langle \vec{v}_i, \vec{v}_j \right\rangle \right)}
        \forall i, j \text{ in token pairs of the vocabulary}

    :param embeddings_list: List with all the token embeddings of the dataset
    :type embeddings_list: List[np.array]
    :return: Result of the formula
    :rtype: float
    """
    all_combinations = combinations(embeddings_list, 2)
    total_num = []
    total_dem = []
    for v1, v2 in all_combinations:
        norm_sum = np.linalg.norm(v1) ** 2 + np.linalg.norm(v2) ** 2
        dem = norm_sum - np.dot(v1, v2)
        total_num.append(norm_sum)
        total_dem.append(dem)
    return np.average(total_num) / np.average(total_dem)
