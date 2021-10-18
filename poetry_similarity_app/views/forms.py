# -*- coding: utf-8 -*-
"""Forms."""
from flask_wtf import FlaskForm
from wtforms import RadioField
from wtforms import SelectField
from wtforms import TextAreaField
from wtforms.validators import InputRequired


class FreeTextForm(FlaskForm):
    text_area = TextAreaField('', [InputRequired()])


class SearchForm(FreeTextForm):
    choices = [('author', 'Poet name'),
               ('poem_title', 'Title of the poem'),
               ('text', 'Excerpt of the poem')]
    select = SelectField('Search for poems:', choices=choices)


class VectorSelectionForm(FlaskForm):
    choices = [("roberta-alberti_poetry_lyrics", "Alberti embeddings"),
               ("roberta-m-poetry-stanzas", "Roberta embeddings cos_sim_sum sum"),
               ("roberta-m-poetry-stanzas-ICM", "Roberta embeddings")
               ]
    radio = RadioField("Similarity based on", choices=choices)
