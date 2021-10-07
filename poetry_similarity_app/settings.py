# -*- coding: utf-8 -*-
"""Application configuration.

Most configuration is set via environment variables.

For local development, use a .env file to set
environment variables.
"""
from environs import Env

env = Env()
env.read_env()

ENV = env.str("FLASK_ENV", default="production")
DEBUG = ENV == "development"
SECRET_KEY = env.str("SECRET_KEY")
BCRYPT_LOG_ROUNDS = env.int("BCRYPT_LOG_ROUNDS", default=13)
DEBUG_TB_ENABLED = DEBUG
DEBUG_TB_INTERCEPT_REDIRECTS = False
CACHE_TYPE = "simple"  # Can be "memcached", "redis", etc.
ES_HOST = env.str("ES_HOST", default='http://localhost')
ES_PORT = env.str("ES_PORT", default='9200')
ES_INDEX = env.str("ES_INDEX", default='roberta-alberti_poetry_lyrics')
URL_PREFIX = env.str("URL_PREFIX", default='/similarity')
