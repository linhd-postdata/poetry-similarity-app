# from flask import Blueprint
# from flask import redirect
# from flask import render_template
# from flask import request
# from flask import url_for
#
# from poetry_similarity_app.views.forms import FreeTextForm
# from poetry_similarity_app.views.forms import VectorSelectionForm
#
# blueprint = Blueprint("insert", __name__, static_folder="../static")
#
#
# @blueprint.route('/insert/', methods=['GET', 'POST'])
# def introducir():
#     form = FreeTextForm()
#     form1 = VectorSelectionForm(prefix="form1")
#     form2 = VectorSelectionForm(prefix="form2")
#     if request.method == 'POST':
#         text = form.data['text_area']
#         vector_selection = "_".join(
#             sorted([key for (key, value) in form1.data.items() if
#                     value is True]))
#         vector_selection2 = "_".join(
#             sorted([key for (key, value) in form2.data.items() if
#                     value is True]))
#         return redirect(url_for('visualize.visualize', text=text,
#                                 selection1=vector_selection,
#                                 selection2=vector_selection2))
#     return render_template('insert/layout.html', form=form, form1=form1, form2=form2)
