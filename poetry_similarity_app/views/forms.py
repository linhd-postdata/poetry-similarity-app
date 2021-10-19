# -*- coding: utf-8 -*-
"""Forms."""
from flask_wtf import FlaskForm
from wtforms import RadioField
from wtforms import SelectField
from wtforms import TextAreaField
from wtforms.validators import InputRequired
from wtforms.form import Form


class FreeTextForm(FlaskForm):
    text_area = TextAreaField('', [InputRequired()])


class SearchForm(FreeTextForm):
    choices = [('author', 'Poet name'),
               ('poem_title', 'Title of the poem'),
               ('text', 'Excerpt of the poem')]
    select = SelectField('Search for poems:', choices=choices)


class VectorSelectionForm(Form):
    choices = [("roberta-alberti_poetry_lyrics", "Alberti embeddings"),
               ("roberta-m_poetry_lyrics", "Roberta embeddings cos_sim_sum sum"),
               ("roberta-m_poetry_stanzas", "Roberta embeddings (ICM)")
               ]
    radio = RadioField("Similarity based on", validators=[InputRequired()],
                       choices=choices)
