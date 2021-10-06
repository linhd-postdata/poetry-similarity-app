from flask import Blueprint
from flask import current_app
from flask import flash
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for


from poetry_similarity_app.utils import query_es
from poetry_similarity_app.views.forms import SearchForm
from poetry_similarity_app.views.forms import VectorSelectionForm

blueprint = Blueprint("search", __name__, static_folder="../static")


@blueprint.route('/search/', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if request.method == 'POST':
        search_string = form.data['text_area']
        select = form.data['select']
        q = f"{select}:{search_string}"
        return redirect(url_for('.results', q=q))
    return render_template('search/layout.html', form=form)


@blueprint.route('/results', methods=['GET', 'POST'])
def results():
    q = request.args['q']
    response = query_es(q)
    # current_app.logger.debug(response)
    form = VectorSelectionForm(prefix="form")
    if request.method == 'POST':
        if 'results' in request.form:
            song_id = request.form['results']
            similarity_base = request.form['form-radio']
            # vector_selection = "_".join(
            #     sorted([key for (key, value) in form.data.items() if
            #             value is True]))
            # current_app.logger.info(f"Formulario 1: {vector_selection}")
            return redirect(url_for('visualize.visualize', num=int(song_id),
                                     similarity_base=similarity_base))
        flash("No song selected, please select a song to visualize", 'warning')
    if response["hits"]["total"]["value"] == 0:
        flash("The search didn't return any results", 'error')
    return render_template('search/results.html', form=form,
                           response=response)
