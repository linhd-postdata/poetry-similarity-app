from flask import Blueprint, flash
from flask import render_template
from flask import request

from poetry_similarity_app.utils import query_es, get_similar_es


blueprint = Blueprint("visualize", __name__, static_folder="../static")


@blueprint.route("/poem", methods=['GET'])
def visualize(n_res=10):
    similarity_base = request.args['similarity_base']
    num = request.args['num']
    if type(num) == str:
        num = int(num)
    song = query_es(q=f"song_id:{num}")
    song = song["hits"]["hits"][0]
    similar_poems = get_similar_es(song["_source"], "ind_rl")
    if similar_poems["hits"]["total"] == 0:
        flash("Similar poems not found in our catalogue",
              'error')
    similar_poems = similar_poems["hits"]["hits"][1:]
    return render_template("visualize/compare.html", song=song,
                           similar_poems=similar_poems, selection=similarity_base)
