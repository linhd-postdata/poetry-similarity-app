# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""
import json

import requests
from flask import flash, current_app, jsonify
from elasticsearch import Elasticsearch

from .settings import ES_HOST
from .settings import ES_INDEX

ES = Elasticsearch([{'host': 'localhost', 'port': 9200}])


def flash_errors(form, category="warning"):
    """Flash all errors for a form."""
    for field, errors in form.errors.items():
        for error in errors:
            flash(f"{getattr(form, field).label.text} - {error}", category)


def query_es(q):
    """Get the ElasticSearch response to the query.
        Query can be on `artist`, `title` or `lyrics`.

    :param q: query to be executed
    :type q: str
    :return: :class:`Response` object
    :rtype: Response
    """
    params = (
        ('pretty', ''),
        ('q', q),
    )
    response = requests.get(f"{ES_HOST}/{ES_INDEX}/_search", params=params)
    if response.status_code == 200:
        response = response.json()
    else:
        response = None
    return response


def get_similar_es(poem, similarity_base):
    q = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc[params.composition_function])",
                "params": {
                    "query_vector": poem[similarity_base],
                    "composition_function": similarity_base
                }
            }
        }
    }
    body = {"query": q, "size": 10}
    response = ES.search(index=ES_INDEX, body=body)
    return response
