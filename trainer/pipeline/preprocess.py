# -*- coding:utf-8 -*-


import os
import glob
import pickle
import random
from logging import getLogger
from typing import List, Optional, Tuple, Dict

import cv2
import luigi
import numpy as np
import pandas as pd

from pipeline.utils.cap_augmentator import CAPAugmentator
from pipeline.config import PREPROCESS_DIR, VOTT_DIR, CAP_BG_DIR, CAP_TARGET_DIR, CAP_OUTPUT_DIR


logger = getLogger()


class Preprocess(luigi.Task):
    is_shuffle = luigi.BoolParameter()
    random_shuffle_seed = luigi.IntParameter()
    cap_bg_copy_num = luigi.IntParameter()
    cap_object_paste_range = luigi.ListParameter()

    def _setup(self) -> Tuple[str, List, Dict]:
        """01_rawディレクトリに格納された画像ファイルリストを取得する.

        Returns:
            Tuple[str, List, Dict]: [vottのcsvファイルパス, cap拡張の背景画像リスト, {label: image list, ...}]
        """
        vott_csv_path = glob.glob(os.path.join(VOTT_DIR, "*.csv"))
        if len(vott_csv_path) != 1:
            exit(1)

        cap_bg_images = glob.glob(os.path.join(CAP_BG_DIR, "*.jpg"))
        cap_bg_images.extend(glob.glob(os.path.join(CAP_BG_DIR, "*.png")))

        cap_target_label_images = {}
        cap_target_labels = [os.path.basename(l) for l in glob.glob(os.path.join(CAP_TARGET_DIR, "*"))]

        logger.info(f"cap labels : {cap_target_labels}")
        for label in cap_target_labels:
            images = glob.glob(os.path.join(CAP_TARGET_DIR, f"{label}/*.png"))
            cap_target_label_images[label] = images

        return vott_csv_path[0], cap_bg_images, cap_target_label_images

    def run(self):
        logger.info("============================= PREPROCESS START =============================")
        output_csv_list = []

        vott_csv_path, cap_bg_images, cap_target_label2images = self._setup()

        # vott形式のファイルをtflite object-detectorに読み込ませられるcsv形式に変換
        output_csv_list = self._transform_vott(vott_csv_path)

        # cap拡張を行い、拡張後のファイルをtflite object-detectorに読み込ませられるcsv形式に変換
        output_csv_list.extend(self._transform_cap(cap_bg_images, cap_target_label2images))

        # object-detectorが読み込める形式のcsvを出力する.
        df = pd.DataFrame(output_csv_list, columns=None, index=None)
        df.to_csv(os.path.join(PREPROCESS_DIR, 'filelist.csv'), index=None, columns=None, header=None)

        with self.output().open('w') as ofile:
            ofile.write(pickle.dumps(
                os.path.join(PREPROCESS_DIR, 'filelist.csv'),
                protocol=pickle.HIGHEST_PROTOCOL
            ))
        logger.info("============================= PREPROCESS FINISH =============================")

    def output(self):
        return luigi.LocalTarget(os.path.join(PREPROCESS_DIR, '.finish'), format=luigi.format.Nop)

    def _transform_cap(self, bg_image_list: List, target_label2images: Dict):
        output_csv_list = []
        
        # bg_images の画像を bg_copy_num 枚数分拡張する.
        for bg_image_path in bg_image_list:
            bg_image_name = os.path.basename(bg_image_path)

            for i in range(self.cap_bg_copy_num):
                labels, bboxes = [], []

                image, x_range, y_range, h_range = self._calc_cap_parames(bg_image_path)
                out_file_name = bg_image_name.split('.')[0] + "_" + str(i) + ".jpg"
                for label, target_images in target_label2images.items():

                    cap_augmentator = CAPAugmentator(
                        target_images,
                        n_objects_range=self.cap_object_paste_range,
                        h_range=h_range,
                        x_range=x_range,
                        y_range=y_range,
                        coords_format='xyxy'
                    )

                    image, aug_bboxes, _, _ = cap_augmentator(image)
                    
                    for aug_bbox in aug_bboxes:
                        bboxes.append(aug_bbox)
                        labels.append(label)
                    
                width, height = image.shape[1], image.shape[0]
                cv2.imwrite(os.path.join(CAP_OUTPUT_DIR, out_file_name), image)
                for label, bbox in zip(labels, bboxes):
                    xmin, ymin, xmax, ymax = bbox
                    output_csv_list.append([
                        "TRAIN",
                        os.path.join(CAP_OUTPUT_DIR, out_file_name),
                        label,
                        xmin/width, ymin/height, '', '',
                        xmax/width, ymax/height, '', ''
                    ])

        return output_csv_list

    def _transform_vott(self, vott_csv_path: str) -> List:
        """vott形式のフォーマットを変換する.
        Args:
            vott_csv_path (str): vottのcsvファイルのパス

        Returns:
            List: 
        """
        output_csv_list = []
        dataset = pd.read_csv(vott_csv_path)

        # split dataset -> train valid test
        seed = None
        if self.is_shuffle:
            seed = self.random_shuffle_seed
        train_list, valid_list, test_list = self._split_data_list(
            dataset['image'].unique(),
            seed
        )

        output_csv_list.extend(
            self._convert_csv_format(
                "TRAIN", dataset, train_list, VOTT_DIR)
        )
        output_csv_list.extend(
            self._convert_csv_format(
                "VALIDATE", dataset, valid_list, VOTT_DIR)
        )
        output_csv_list.extend(
            self._convert_csv_format(
                "TEST", dataset, test_list, VOTT_DIR)
        )
        return output_csv_list

    def _split_data_list(self, unique_image_list: List, random_shuffle_seed: Optional[int]) -> Tuple[List, List, List]:
        """与えられた画像リストをtrain,valid,testに分割する.

        Args:
            unique_image_list (List): ユニークな画像ファイルリスト
            random_shuffle_seed (Optional[int]): 乱数のシード値

        Returns:
            Tuple[List, List, List]: [train用画像リスト, valid用画像リスト, test用画像リスト]
        """
        train_list, valid_list, test_list = [], [], []
        unique_image_list = sorted(unique_image_list)
        dataset_num = len(unique_image_list)

        # train 60%, validation 20%, test: 20%
        train_num = int(0.6 * dataset_num)
        valid_num = int(0.2 * dataset_num)
        test_num = int(0.2 * dataset_num)

        if random_shuffle_seed is None:
            train_list = unique_image_list[:train_num]
            valid_list = unique_image_list[train_num:train_num+valid_num]
            test_list = unique_image_list[train_num+valid_num:]
        else:
            random.seed(random_shuffle_seed)
            shuffle_image_list = random.sample(unique_image_list, dataset_num)
            train_list = shuffle_image_list[:train_num]
            valid_list = shuffle_image_list[train_num:train_num+valid_num]
            test_list = shuffle_image_list[train_num+valid_num:]

        return train_list, valid_list, test_list

    def _convert_csv_format(self, tag: str, dataframe: pd.DataFrame, target_file_list: List, base_dir: str) -> List:
        """vott形式のフォーマットをobject-detectorが読み込める形式に変換する.

        Args:
            tag (str): TRAIN or VALIDATE or TEST
            dataframe (pd.DataFrame): vott csvのデータフレーム
            target_file_list (List): 対象の画像リスト
            base_dir (str): 画像ファイルのディレクトリパス

        Returns:
            List: object-detectorが読み込める形式のリスト
        """
        result_list = []

        for target_file in target_file_list:
            for _, row in (dataframe[dataframe['image'] == target_file]).iterrows():
                file_path = os.path.join(base_dir, row.image)

                tmp_img = cv2.imread(file_path)
                img_width, img_height = tmp_img.shape[1], tmp_img.shape[0]
                xmin = row.xmin / img_width
                ymin = row.ymin / img_height
                xmax = row.xmax / img_width
                ymax = row.ymax / img_height

                result_list.append(
                    [tag, file_path, row.label, xmin, ymin, '', '', xmax, ymax, '', ''])
        return result_list

    def _calc_cap_parames(self, image_path: str) -> Tuple[np.ndarray, List, List, List]:
        h_range, x_range, y_range = [], [], []

        image = cv2.imread(image_path)

        while True:
            width, height = image.shape[1], image.shape[0]

            min_length = height if height < width else width

            if (min_length // 2) > 200:
                break

            image = cv2.resize(image, (width * 10, height * 10))

        x_range = [10, width]
        y_range = [10, height]
        h_range = [min_length//10, min_length//2]

        return image, x_range, y_range, h_range