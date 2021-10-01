release: flask db upgrade
web: gunicorn poetry_similarity_app.app:create_app\(\) -b 0.0.0.0:$PORT -w 3
