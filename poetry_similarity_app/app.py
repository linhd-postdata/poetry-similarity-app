# -*- coding: utf-8 -*-
"""The app module, containing the app factory function."""
import logging
import sys

from flask import Flask
from flask import render_template

from poetry_similarity_app import commands
from poetry_similarity_app import views
from poetry_similarity_app import extensions
from poetry_similarity_app import settings


def create_app(config_object="poetry_similarity_app.settings"):
    """Create application factory, as explained here: http://flask.pocoo.org/docs/patterns/appfactories/.

    :param config_object: The configuration object to use.
    """
    app = Flask(__name__.split(".")[0], static_folder="./static/")
    app.config.from_object(config_object)
    register_extensions(app)
    register_blueprints(app)
    register_errorhandlers(app)
    register_commands(app)
    configure_logger(app)
    return app


def register_extensions(app):
    """Register Flask extensions."""
    extensions.cache.init_app(app)
    extensions.debug_toolbar.init_app(app)
    extensions.flask_static_digest.init_app(app)
    return None


def register_blueprints(app):
    """Register Flask blueprints."""
    app.register_blueprint(views.home.blueprint, url_prefix=settings.URL_PREFIX)
    app.register_blueprint(views.search.blueprint, url_prefix=settings.URL_PREFIX)
    app.register_blueprint(views.visualize.blueprint, url_prefix=settings.URL_PREFIX)
    # app.register_blueprint(views.insert.blueprint)
    return None


def register_errorhandlers(app):
    """Register error handlers."""

    def render_error(error):
        """Render error template."""
        # If a HTTPException, pull the `code` attribute; default to 500
        error_code = getattr(error, "code", 500)
        return render_template(f"{error_code}.html"), error_code

    for errcode in [401, 404, 500]:
        app.errorhandler(errcode)(render_error)
    return None


def register_commands(app):
    """Register Click commands."""
    app.cli.add_command(commands.test)
    app.cli.add_command(commands.lint)


def configure_logger(app):
    """Configure loggers."""
    handler = logging.StreamHandler(sys.stdout)
    if not app.logger.handlers:
        app.logger.addHandler(handler)
