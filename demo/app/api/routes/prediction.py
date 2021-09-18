# -*- coding:utf-8 -*-
import io
import uuid
from typing import List
from logging import getLogger

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.requests import Request

from app.models.prediction import PredictionResult
from app.core import security
from app.services.models import ObjectDetectionModel

router = APIRouter()

logger = getLogger(__name__)


@router.post("/predict", response_model=PredictionResult, name="predict")
def post_predict(
    request: Request,
    files: List[UploadFile] = File(...),
    authenticated: bool = Depends(security.validate_request),
) -> PredictionResult:
    job_id = str(uuid.uuid4())
    model: ObjectDetectionModel = request.app.state.model
    prediction: PredictionResult = model.predict(
        io.BytesIO(files[0].file.read()))
    logger.info(f"test {job_id}: {prediction}")
    return prediction
