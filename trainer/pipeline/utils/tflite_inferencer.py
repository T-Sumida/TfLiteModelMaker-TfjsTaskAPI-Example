# -*- coding:utf-8 -*-
import os
import copy
from typing import Tuple, List

import cv2
import numpy as np
import tensorflow as tf


def draw_bboxes(image: np.ndarray, bboxes: List, scores: List, classes: List, detect_num: int, score_thr: float = 0.3) -> np.ndarray:
    tmp_image = copy.deepcopy(image)

    for i in range(len(bboxes)):
        score = scores[i]
        bbox = bboxes[i]
        class_id = classes[i].astype(np.int)

        if score < score_thr:
            continue

        x1, y1 = int(bbox[1]), int(bbox[0])
        x2, y2 = int(bbox[3]), int(bbox[2])

        cv2.putText(
            tmp_image, 'ID:' + str(class_id) + ' ' +
            '{:.3f}'.format(score),
            (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            cv2.LINE_AA)
        cv2.rectangle(tmp_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
    return tmp_image


class TfliteInferencer(object):
    def __init__(self, model_path: str) -> None:
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()

        base_file_name = os.path.basename(model_path)
        self.model_type = 'int8' if ('int8' in base_file_name) else 'float16'

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        self.img_size = (
            self.input_details['shape'][1], self.input_details['shape'][2])

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = image[:, :, [2, 1, 0]]  # BGR2RGB
        img = cv2.resize(img, self.img_size)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

        if self.model_type == 'int8':
            img = img.astype(np.uint8)
        elif self.model_type == 'float16':
            img = img.astype(np.float32)
        return img

    def _postprocess(self, input_image: np.ndarray, bbox: np.ndarray, classes: np.ndarray, scores: np.ndarray, object_num: int):
        post_bboxes, post_classes, post_scores = [], [], []

        width, height = input_image.shape[1], input_image.shape[0]

        for i in range(object_num):
            post_classes.append(classes[i].astype(np.int))
            post_scores.append(scores[i])

            x1, y1 = int(bbox[i][1] * width), int(bbox[i][0] * height)
            x2, y2 = int(bbox[i][3] * width), int(bbox[i][2] * height)
            post_bboxes.append([x1, y1, x2, y2])

        return post_bboxes, post_classes, post_scores, object_num

    def inference(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        img = self._preprocess(image)

        self.interpreter.set_tensor(self.input_details['index'], img)
        self.interpreter.invoke()

        bboxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        num = self.interpreter.get_tensor(self.output_details[3]['index'])

        return self._postprocess(image, np.squeeze(bboxes), np.squeeze(classes), np.squeeze(scores), int(num[0]))
