# -*- coding:utf-8 -*-
import os
import glob
import shutil
import hashlib
import configparser
from logging import getLogger

import luigi
import numpy as np
from luigi.configuration import get_config

import pipeline
from pipeline.config import VOTT_DIR, CAP_TARGET_DIR, CAP_BG_DIR, PREPROCESS_DIR, MODEL_DIR, REPORT_DIR


logger = getLogger()

TMP_DIR = ".tmp"


def get_dataset_hash() -> str:
    """データセットディレクトリのファイルパスからハッシュを作成する。

    Returns:
        str: データセットのハッシュ値
    """
    files = []
    files.extend(glob.glob(os.path.join(VOTT_DIR, "*")))
    files.extend(glob.glob(os.path.join(CAP_TARGET_DIR, "**"), recursive=True))
    files.extend(glob.glob(os.path.join(CAP_BG_DIR, "**"), recursive=True))

    return hashlib.sha512(','.join(files).encode()).hexdigest()


def remove_luigi_checkpoint(dir_path: str):
    """チェックポイントファイルを削除する。

    Args:
        dir_path (str): ディレクトリパス
    """
    if os.path.exists(os.path.join(dir_path, ".finish")):
        os.remove(os.path.join(dir_path, ".finish"))


def check_update():
    """前回実行時とのデータセット・パラメータを比較する。変更があった場合、各タスクの出力ディレクトリからチェックポイントファイルを削除する。
    """
    COMPONENT_NAMES = ["Preprocess", "Train", "Report"]
    OUTPUT_PATHS = [PREPROCESS_DIR, MODEL_DIR, REPORT_DIR]
    checkpoint_delete_flag = False

    dataset_hash = get_dataset_hash()
    if not os.path.exists(TMP_DIR):
        # 比較対象がない場合は保存のみ
        os.makedirs(TMP_DIR)
        shutil.copy('./conf/param.ini', TMP_DIR)
        with open(os.path.join(TMP_DIR, "dataset_hash.txt"), 'w') as f:
            f.write(dataset_hash)
        
        for output_path in OUTPUT_PATHS:
            remove_luigi_checkpoint(output_path)
        return

    # datasetを比較
    with open(os.path.join(TMP_DIR, "dataset_hash.txt"), 'r') as f:
        if dataset_hash != f.read():
            # datasetの中身が前回と異なる場合、checkpoint_delete_flag をTrueにする。
            checkpoint_delete_flag = True
            logger.info("== Dataset Changed ==")

    # 前回のparam.iniと今回のparam.iniの中身を比較する。
    before_param = configparser.ConfigParser()
    before_param.read(os.path.join(TMP_DIR, "param.ini"), encoding='utf-8')

    for component_name, output_path in zip(COMPONENT_NAMES, OUTPUT_PATHS):
        if checkpoint_delete_flag:
            remove_luigi_checkpoint(output_path)
            continue

        config_params = dict(get_config().items(component_name))
        for k, v in config_params.items():
            if v != before_param[component_name].get(k):
                remove_luigi_checkpoint(output_path)
                checkpoint_delete_flag = True
                logger.info(f"== {component_name} Changed ==")

    # 最後にparam.iniとdataset_hashを更新する。
    shutil.copy('./conf/param.ini', TMP_DIR)
    with open(os.path.join(TMP_DIR, "dataset_hash.txt"), 'w') as f:
        f.write(dataset_hash)


if __name__ == "__main__":
    luigi.configuration.LuigiConfigParser.add_config_path('./conf/param.ini')
    check_update()

    np.random.seed(57)
    luigi.run()
