# -*- coding:utf-8 -*-
import os
import pickle
from typing import Dict
from logging import getLogger

import cv2
import luigi
import mlflow
import pandas as pd
from luigi.configuration import get_config

from pipeline import Train, Preprocess
from pipeline.utils.tflite_inferencer import TfliteInferencer, draw_bboxes
from pipeline.config import REPORT_DIR, PREPROCESS_DIR, MODEL_DIR

logger = getLogger(__name__)

HEADER = ["KIND", "path", "label", "x1",
          "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


class Report(luigi.Task):
    run_name = luigi.Parameter(default="exp")

    def requires(self):
        return {"model": Train(), "filelist": Preprocess()}

    def run(self):
        logger.info("============================= REPORTER START =============================")
        with self.input()['model'].open('rb') as f:
            train_result: Dict = pickle.load(f)
        with self.input()['filelist'].open('rb') as f:
            csv_path: str = pickle.load(f)

        self._test_inference(csv_path, train_result['model_path']['int8'], train_result['model_path']['float16'])
        with mlflow.start_run(run_name=self.run_name) as run_obj:
            self._record_params()
            self._record_metrics("valid", train_result['valid_metrics'])
            self._record_metrics("test", train_result['test_metrics'])

            mlflow.log_artifacts(PREPROCESS_DIR)
            mlflow.log_artifacts(MODEL_DIR)
            mlflow.log_artifacts(REPORT_DIR)
        self.output().open('w').close()
        logger.info("============================= REPORTER FINISH =============================")

    def _test_inference(self, csv_path: str, int8_model_path: str, float16_model_path: str):
        df = pd.read_csv(csv_path, names=HEADER)
        test_df = df[df['KIND'] == "TEST"]
        test_files = list(test_df['path'].unique())

        int8_model = TfliteInferencer(int8_model_path)
        float16_model = TfliteInferencer(float16_model_path)

        for i, file_path in enumerate(test_files):
            if i == 10:
                break

            img = cv2.imread(file_path)

            u8_bboxes, u8_classes, u8_scores, u8_obj_num = int8_model.inference(img)
            f16_bboxes, f16_classes, f16_scores, f16_obj_num = float16_model.inference(img)

            u8_result_img = draw_bboxes(img, u8_bboxes, u8_classes, u8_scores, u8_obj_num)
            f16_result_img = draw_bboxes(img, f16_bboxes, f16_classes, f16_scores, f16_obj_num)

            cv2.imwrite(os.path.join(REPORT_DIR, f"{i}_uint8.jpg"), u8_result_img)
            cv2.imwrite(os.path.join(REPORT_DIR, f"{i}_float16.jpg"), f16_result_img)

    def _record_params(self):
        preprocess_params = dict(get_config().items('Preprocess'))
        train_params = dict(get_config().items('Train'))

        for k, v in preprocess_params.items():
            mlflow.log_param(k, v)

        for k, v in train_params.items():
            mlflow.log_param(k, v)

    def _record_metrics(self, key_prefix: str, metrics: Dict):
        for k, v in metrics.items():
            mlflow.log_metric(key_prefix + "_" + k, v)

    def output(self):
        return luigi.LocalTarget(os.path.join(REPORT_DIR, '.finish'), format=luigi.format.Nop)
