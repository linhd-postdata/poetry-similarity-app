from flask import Blueprint, flash
from flask import render_template
from flask import request

from poetry_similarity_app.utils import query_es, get_similar_es, \
    add_poem_metadata
from poetry_similarity_app.views.models import MODELS

blueprint = Blueprint("visualize", __name__, static_folder="../static")


@blueprint.route("/poem", methods=['GET'])
def visualize(n_res=10):
    similarity_base = request.args['similarity_base']
    num = request.args['num']
    if type(num) == str:
        num = int(num)
    poem = query_es(q=f"song_id:{num}", index=similarity_base)
    poem = poem["hits"]["hits"][0]
    add_poem_metadata(poem)
    similar_poems = get_similar_es(poem["_source"], similarity_base)
    if similar_poems["hits"]["total"] == 0:
        flash("Similar poems not found in our catalogue",
              'error')
    similar_poems = similar_poems["hits"]["hits"][1:n_res]
    for similar_p in similar_poems:
        add_poem_metadata(similar_p)
    return render_template("visualize/compare.html", poem=poem,
                           similar_poems=similar_poems,
                           selection=MODELS[similarity_base])
