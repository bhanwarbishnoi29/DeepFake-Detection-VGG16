"""
Celeb-DF DeepFake Detection System package initialization.
Configures environment defaults for Keras and Matplotlib caching.
"""

import os

os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
