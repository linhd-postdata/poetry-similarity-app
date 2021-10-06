"""Settings module for test app."""
ENV = "development"
TESTING = True
SECRET_KEY = "not-so-secret-in-tests"
# For faster tests; needs at least 4 to avoid "ValueError: Invalid rounds"
BCRYPT_LOG_ROUNDS = 4
DEBUG_TB_ENABLED = False
CACHE_TYPE = "simple"  # Can be "memcached", "redis", etc.
WTF_CSRF_ENABLED = False  # Allows form testing
