# -*- coding:utf-8 -*-

from typing import Callable
from logging import getLogger

from fastapi import FastAPI

from app.core.config import MODEL_PATH, NUM_THREADS
from app.services.models import ObjectDetectionModel


logger = getLogger(__name__)


def _startup_model(app: FastAPI) -> None:
    model_instalce = ObjectDetectionModel(MODEL_PATH, NUM_THREADS)
    app.state.model = model_instalce


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        _startup_model(app)
    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        _shutdown_model(app)
    return shutdown
