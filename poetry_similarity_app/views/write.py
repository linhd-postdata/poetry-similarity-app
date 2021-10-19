from flask import Blueprint, flash
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for

from poetry_similarity_app.views.forms import FreeTextForm
from poetry_similarity_app.views.forms import VectorSelectionForm

blueprint = Blueprint("write", __name__, static_folder="../static")


@blueprint.route('/write/', methods=['GET', 'POST'])
def write_poem():
    form = FreeTextForm()
    form1 = VectorSelectionForm(prefix="form")
    if request.method == 'POST':
        if 'form-radio' in request.form:
            text = form.data['text_area']
            similarity_base = request.form['form-radio']
            return redirect(url_for('visualize.visualize_text', text=text,
                                    similarity_base=similarity_base))
        # flash("No similarity based selected", "warning")
    return render_template('write/layout.html', form=form, form1=form1)
