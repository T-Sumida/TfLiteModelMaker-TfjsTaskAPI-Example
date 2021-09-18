# -*- coding:utf-8 -*-
import os
from logging import getLogger
from typing import BinaryIO, Tuple

import cv2
import numpy as np
import tensorflow as tf

from app.models.prediction import PredictionResult


logger = getLogger(__name__)


class ObjectDetectionModel(object):
    def __init__(self, model_path: str, num_threads: int = 1) -> None:
        """init
        Args:
            tracking_url (str): mlflow uri
            model_name (str): model name of mlflow
            size (int): target image size
        """
        self._load_model(model_path, num_threads)

    def _load_model(self, model_path: str, num_threads: int) -> None:
        """load model"""
        logger.info(f"load model in {model_path}")
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()

        base_file_name = os.path.basename(model_path)
        self.model_type = 'int8' if ('int8' in base_file_name) else 'float16'

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        self.img_size = (
            self.input_details['shape'][1], self.input_details['shape'][2])
        logger.info("initialized model")

    def _pre_process(self, bin_data: BinaryIO) -> np.array:
        """preprocess
        Args:
            bin_data (BinaryIO): binary image data
        Returns:
            np.array: image data
        """
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        width, height = img.shape[1], img.shape[0]
        img = cv2.resize(img, self.img_size)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

        if self.model_type == 'int8':
            img = img.astype(np.uint8)
        elif self.model_type == 'float16':
            img = img.astype(np.float32)
        return img, width, height

    def _post_process(self, width: int, height: int, bboxes: np.ndarray, classes: np.ndarray, scores: np.ndarray, object_num: int) -> PredictionResult:
        """post process
        Args:
            prediction (np.array): result of predict
        Returns:
           PredictionResult: prediction
        """
        post_bboxes, post_classes, post_scores = [], [], []

        for i in range(object_num):
            post_classes.append(int(classes[i]))
            post_scores.append(float(scores[i]))

            x1, y1 = int(bboxes[i][1] * width), int(bboxes[i][0] * height)
            x2, y2 = int(bboxes[i][3] * width), int(bboxes[i][2] * height)
            post_bboxes.append([x1, y1, x2, y2])

        dcp = PredictionResult(bboxes=post_bboxes, scores=post_scores, classes=post_classes)
        return dcp

    def _predict(self, img: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """predict
        Args:
            img (np.array): image data
        Returns:
            np.array: prediction
        """
        self.interpreter.set_tensor(self.input_details['index'], img)
        self.interpreter.invoke()

        bboxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        num = self.interpreter.get_tensor(self.output_details[3]['index'])
        return np.squeeze(bboxes), np.squeeze(classes), np.squeeze(scores), int(num[0])

    def predict(self, bin_data: BinaryIO) -> PredictionResult:
        """predict method
        Args:
            bin_data (BinaryIO): binary image data
        Returns:
            PredictionResult: prediction
        """
        img, width, height = self._pre_process(bin_data)
        bboxes, classes, scores, obj_num = self._predict(img)
        post_processed_result = self._post_process(
            width, height, bboxes, classes, scores, obj_num)
        return post_processed_result
