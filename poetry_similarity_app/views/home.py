from flask import Blueprint
from flask import render_template

blueprint = Blueprint("home", __name__, static_folder="../static")


@blueprint.route("/", methods=["GET"])
def home():
    """Home page."""
    return render_template("home/index.html")


@blueprint.route("/about")
def about():
    """About page."""
    return render_template("home/about.html")

