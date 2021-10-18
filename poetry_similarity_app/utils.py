# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""

import requests
from elasticsearch import Elasticsearch
from flask import flash

from .settings import ES_HOST
from .settings import ES_INDEX
from .settings import ES_PORT
from .views.models import MODELS

ES = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT}])


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
                "source": "cosineSimilarity(params.query_vector, doc[params.composition_function])",
                "params": {
                    "query_vector": poem[MODELS[similarity_base]["similarity_fn"]],
                    "composition_function": MODELS[similarity_base]["similarity_fn"]
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
