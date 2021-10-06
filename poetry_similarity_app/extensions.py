# -*- coding: utf-8 -*-
"""Extensions module.

Each extension is initialized in the app factory located in app.py.
"""
from flask_caching import Cache
from flask_debugtoolbar import DebugToolbarExtension
from flask_static_digest import FlaskStaticDigest


cache = Cache()
debug_toolbar = DebugToolbarExtension()
flask_static_digest = FlaskStaticDigest()
