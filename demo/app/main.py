# -*- coding:utf-8 -*-
from logging import getLogger

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.front.router import front_router
from app.api.routes.router import api_router
from app.core.config import (API_PREFIX, APP_NAME, APP_VERSION, IS_DEBUG)
from app.core.event_handler import (start_app_handler, stop_app_handler)

logger = getLogger(__name__)


def get_app() -> FastAPI:
    app = FastAPI(title=APP_NAME, version=APP_VERSION, debug=IS_DEBUG)
    app.include_router(api_router, prefix=API_PREFIX)
    app.include_router(front_router, prefix="")

    app.add_event_handler("startup", start_app_handler(app))
    app.add_event_handler("shutdown", stop_app_handler(app))
    logger.info("start.")

    return app


app = get_app()
app.mount("/static", StaticFiles(directory="static"), name="static")