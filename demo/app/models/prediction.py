# -*- coding:utf-8 -*-
from typing import List
from pydantic import BaseModel


class PredictionResult(BaseModel):
    bboxes: List
    scores: List
    classes: List
