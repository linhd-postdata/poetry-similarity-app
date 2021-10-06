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
    choices = [('artist_name', 'Poet name'),
               ('track_name', 'Title of the poem'),
               ('text', 'Excerpt of the poem')]
    select = SelectField('Search for poems:', choices=choices)


class VectorSelectionForm(FlaskForm):
    choices = [("alberti", "Alberti embeddings"),
               # ("spacy", "Spacy embeddings"),
               # ("bert", "Bert embeddings")
               ]
    radio = RadioField("Similarity based on", choices=choices)
