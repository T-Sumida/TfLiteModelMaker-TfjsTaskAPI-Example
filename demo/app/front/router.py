# -*- coding:utf-8 -*-

from fastapi import APIRouter, Request
from starlette.templating import _TemplateResponse
from fastapi.templating import Jinja2Templates

# templates配下に格納したindex.htmlをrenderするために必要
templates = Jinja2Templates(directory="templates")


front_router = APIRouter()


@front_router.get("/")
async def index(request: Request) -> _TemplateResponse:
    return templates.TemplateResponse("tflite.html", {"request": request})


@front_router.get("/webcam")
async def webcam(request: Request) -> _TemplateResponse:
    return templates.TemplateResponse("tflite-webcam.html", {"request": request})


@front_router.get("/webapi")
async def webapi(request: Request) -> _TemplateResponse:
    return templates.TemplateResponse("tflite-api.html", {"request": request})
