# -*- coding:utf-8 -*-
from starlette.config import Config
from starlette.datastructures import Secret

APP_VERSION = "0.0.1"
APP_NAME = "TfLite Object-Detector Example"
API_PREFIX = "/api"

config = Config(".env")

API_KEY: Secret = config("API_KEY", cast=Secret)
IS_DEBUG: bool = config("IS_DEBUG", cast=bool, default=False)

MODEL_PATH: str = config("MODEL_PATH", cast=str)
NUM_THREADS: int = config("NUM_THREADS", cast=int)
