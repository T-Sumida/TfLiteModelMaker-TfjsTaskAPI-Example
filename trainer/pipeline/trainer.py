# -*- coding:utf-8 -*-
import os
import pickle
from logging import getLogger

import luigi
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import QuantizationConfig

from pipeline import Preprocess
from pipeline.config import MODEL_DIR


logger = getLogger(__name__)


class Train(luigi.Task):
    model_name = luigi.Parameter(default="efficientdet_lite0")
    batch_size = luigi.IntParameter(default=8)
    epochs = luigi.IntParameter(default=50)
    train_whole_model = luigi.BoolParameter()

    def requires(self):
        return {"filelist": Preprocess()}

    def run(self):
        logger.info("============================= TRAINER START =============================")
        with self.input()['filelist'].open('rb') as f:
            csv_path: str = pickle.load(f)

        train_data, valid_data, test_data = object_detector.DataLoader.from_csv(
            csv_path)
        
        logger.info("LOAD DATASET")

        spec = model_spec.get(self.model_name)
        model = object_detector.create(
            train_data,
            model_spec=spec,
            validation_data=valid_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            train_whole_model=self.train_whole_model
        )
        logger.info("TRAINING FINISH")

        valid_metrics = model.evaluate(valid_data)
        test_metrics = model.evaluate(test_data)

        logger.info("METRICS FINISH")

        model.export(export_dir=MODEL_DIR, tflite_filename='model_int8.tflite')
        config = QuantizationConfig.for_float16()
        model.export(export_dir=MODEL_DIR, tflite_filename='model_fp16.tflite', quantization_config=config)

        logger.info("EXPORT MODEL")

        with self.output().open('w') as ofile:
            ofile.write(pickle.dumps(
                {
                    'model_path': {
                        'int8': os.path.join(MODEL_DIR, 'model_int8.tflite'),
                        'float16': os.path.join(MODEL_DIR, 'model_fp16.tflite'),
                    },
                    'valid_metrics': valid_metrics,
                    'test_metrics': test_metrics
                },
                protocol=pickle.HIGHEST_PROTOCOL
            ))
        
        logger.info("============================= TRAINER FINISH =============================")

    def output(self):
        return luigi.LocalTarget(os.path.join(MODEL_DIR, '.finish'), format=luigi.format.Nop)
