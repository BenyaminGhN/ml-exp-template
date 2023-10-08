import os 
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from omegaconf import DictConfig

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
import cv2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

from .preprocessing import SingleImagePreprocessor, SingleImagePreprocessorSeg


class DataLoader():
    def __init__(self, config: DictConfig, data_dir):
        self.config_ = config
        self.config = config.data_pipeline
        self.data_dir = data_dir
        self.class_weights = None
        self.is_seg = self.config.is_seg
        
        np.random.seed(self.config_.seed)

    def get_class_weights(self):
        return self.class_weights

    def create_train_val_generator(self):
        if self.is_seg:
            csv_filename = self.config_.target_csv
        else:
            csv_filename = self.config_.target_csv
        train_df, val_df = self._get_train_val_df(os.path.join(self.data_dir, csv_filename))

        # Train
        train_seq = self._create_tf_seq(
            train_df, augmentation=self.config.to_augment, shuffle=self.config.shuffle
        )
        n_iter_train = len(train_df) // self.config.batch_size

        # Validation
        val_seq = self._create_tf_seq(val_df)
        n_iter_val = len(val_df) // self.config.batch_size

        return train_seq, n_iter_train, val_seq, n_iter_val
    
    def create_eval_generator(self):
        if self.is_seg:
            csv_filename = self.config_.target_csv
        else:
            csv_filename = self.config_.target_csv
        csv_file_path = os.path.join(self.data_dir, csv_filename)

        if Path(csv_file_path).exists():
            eval_df = self._get_eval_df(str(csv_file_path))
        else:
            dicom_list = Path(self.data_dir).rglob("*.dcm")
            if self.is_seg:
                labels = {"SOPInstanceUID": [], "Path": [], "MaskPath": []}
                for dcm in dicom_list:
                    print(dcm)
                    labels["SOPInstanceUID"].append(str(dcm.stem))
                    labels["Path"].append(str(dcm))
                    labels["MaskPath"].append("")
            else:
                labels = {"SOPInstanceUID": [], "Path": [], "NumLabels": []}
                for dcm in dicom_list:
                    labels["SOPInstanceUID"].append(str(dcm.stem))
                    labels["Path"].append(str(dcm.name))
                    labels["NumLabels"].append(0)

            eval_df = pd.DataFrame(labels)

        # Evaluation
        eval_seq = self._create_tf_seq(eval_df, shuffle_on_epoch_end=False)
        n_iter_eval = len(eval_df) // self.config.batch_size

        return eval_seq, n_iter_eval
    
    def _get_eval_df(self, path):
        df = pd.read_csv(path)
        df = df[df['Split']=='evaluation']
        if not self.is_seg:
            df['NumLabels'] = df['Label'].apply(lambda x: 1 if x=='abnormal' else 0)

        return df

    def _get_train_val_df(self, path):

        df = pd.read_csv(path)

        df = df[df['Split']=='train']

        if not self.is_seg:
            df['NumLabels'] = df['Label'].apply(lambda x: 1 if x=='abnormal' else 0)
        n_data = len(df)

        val_split = self.config.val_split
        val_ind = np.random.randint(0, n_data, int(n_data * val_split))

        val_df = df.iloc[val_ind]
        train_df = df.drop(val_df.index)

        do_oversampling = self.config_.augmentation.oversampling
        if do_oversampling:
            abnormal_train = train_df[train_df['Label']=='abnormal']
            for i in range(3):
                train_df = train_df.append(abnormal_train, ignore_index=True)
            
        train_df = shuffle(train_df, random_state=self.config_.seed)
        val_df = shuffle(val_df, random_state=self.config_.seed)

        if not self.is_seg:
            if self.config.is_binary:
                y_ = train_df['NumLabels'].values
                y_r = 1-y_
                gt_arr = np.vstack([y_r, y_]).T
            else:    
                gt_arr = train_df['NumLabels'].values
            self.class_weights = self.__calculating_class_weights(gt_arr)

        return train_df, val_df

    @staticmethod
    def __calculating_class_weights(y_true):
        if len(np.shape(y_true)) == 1:
            y_true = y_true.reshape((-1, 1))
        number_dim = np.shape(y_true)[1]
        weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            weights[i] = compute_class_weight('balanced', classes=np.unique(y_true[:, i]), y=y_true[:, i])
        return weights

    def _create_tf_seq(self, df, augmentation=False, 
                       shuffle=False, shuffle_on_epoch_end=True):
        
        if shuffle:
            permutation = np.random.RandomState(self.config_.seed).permutation(len(df))
            dataset_csv_file = df.iloc[permutation].reset_index(drop=True)
        else:
            dataset_csv_file = df
        image_source_dir = self.data_dir
        target_size = self.config_.preprocessing.target_size
        class_names = self.config.class_names
        is_binary = self.config.is_binary
        batch_size = self.config.batch_size
        add_channel = self.config_.preprocessing.add_channel
        crop_ratio = self.config_.preprocessing.crop_ratio
        steps = int(np.ceil(len(df) / batch_size))

        x_col = "Path"
        if self.is_seg:
            y_col = "MaskPath"
            df_seq = AugmentedImageSequenceSeg(
            dataset_csv_file=dataset_csv_file,
            x_col=x_col,
            y_col=y_col,
            class_names=class_names,
            source_image_dir=image_source_dir,
            batch_size=batch_size,
            target_size=target_size,
            add_channel=add_channel,
            augmentation=augmentation,
            steps=steps,
            shuffle_on_epoch_end=shuffle_on_epoch_end,
            random_state=self.config_.seed,
        )
        else:
            y_col = "NumLabels"
            df_seq = AugmentedImageSequence(
            dataset_csv_file=dataset_csv_file,
            x_col=x_col,
            y_col=y_col,
            class_names=class_names,
            source_image_dir=image_source_dir,
            is_binary=is_binary,
            batch_size=batch_size,
            target_size=target_size,
            add_channel = add_channel,
            crop_ratio=crop_ratio, 
            augmentation=augmentation,
            steps=steps,
            shuffle_on_epoch_end=shuffle_on_epoch_end,
            random_state=self.config_.seed,
        )

        return df_seq
        

class AugmentedImageSequence(Sequence):
    def __init__(
        self,
        dataset_csv_file,
        x_col,
        y_col,
        class_names,
        source_image_dir,
        is_binary=False,
        batch_size=16,
        target_size=(224, 224),
        add_channel=True,
        crop_ratio=[0.0, 0.0], 
        augmentation=True,
        verbose=0,
        steps=None,
        shuffle_on_epoch_end=True,
        random_state=None,
    ):

        self.dataset_df = dataset_csv_file
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.add_channel = add_channel
        self.crop_ratio = crop_ratio
        self.augmentation = augmentation
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.x_col = x_col
        self.y_col = y_col
        self.is_binary = is_binary
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def get_dataset_df(self):
        return self.dataset_df
    
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)

        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_y

    def load_image(self, image_file):
        image_path = os.path.join(str(self.source_image_dir), image_file)

        image_array = self.preprocess_img(image_path)

        return image_array

    def transform_batch_images(self, batch_x):
        if self.augmentation:
            sometimes = lambda aug: iaa.Sometimes(0.25, aug)
            augmenter = iaa.Sequential(
                [
                    iaa.Fliplr(0.25),
                    iaa.Flipud(0.10),
                    iaa.geometric.Affine(
                        rotate=(-45, 45), order=1, mode="constant", fit_output=False
                    ),
                    sometimes(
                        iaa.Crop(px=(0, 25), keep_size=True, sample_independently=False)
                    ),
                ],
                random_order=True,
            )
            batch_x = augmenter.augment_images(batch_x)
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError(
                """
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """
            )
        return self.y[: self.steps * self.batch_size]

    def get_x_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError(
                """
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """
            )
        return self.x_path[: self.steps * self.batch_size]

    def prepare_dataset(self):
        # df = self.dataset_df.sample(frac=1.0, random_state=self.random_state)
        df = self.dataset_df
        self.x_path, self.y = (df[self.x_col].values, 
                                df[self.y_col].values.astype('float32'))
        
        if self.is_binary:
            y_ = self.y
            y_r = 1-y_
            self.y = np.vstack([y_r, y_]).T

        # self.y = np.where(self.y=='abnormal', 1, 0)

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

    def preprocess_img(self, path):
        preprocessor = SingleImagePreprocessor(target_size=self.target_size,
                                                add_channel=self.add_channel,
                                                crop_ratio=self.crop_ratio)
        img = preprocessor.read_dcm(path)
        preprocessed_img = preprocessor.preprocess_image(img)
        return preprocessed_img

class AugmentedImageSequenceSeg(Sequence):
    def __init__(
        self,
        dataset_csv_file,
        x_col,
        y_col,
        class_names,
        source_image_dir,
        is_binary=False, 
        batch_size=16,
        target_size=(224, 224),
        add_channel=True,
        augmentation=True,
        verbose=0,
        steps=None,
        shuffle_on_epoch_end=True,
        random_state=None,
    ):

        self.dataset_df = dataset_csv_file
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.add_channel = add_channel
        self.is_binary = is_binary,
        self.augmentation = augmentation
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.x_col = x_col
        self.y_col = y_col
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def get_dataset_df(self):
        return self.dataset_df
    
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y_path = self.y_path[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch = [self.load_image(x_path, y_path) for x_path, y_path in zip(batch_x_path, batch_y_path)]

        batch_x = []
        batch_y = []
        for x, y in batch:
            batch_x.append(x)
            batch_y.append(y)
        batch_x, batch_y = np.array(batch_x), np.array(batch_y)
        batch_x = self.transform_batch_images(batch_x)

        return batch_x, batch_y

    def load_image(self, image_file, mask_path):
        # image_path = os.path.join(str(self.source_image_dir), image_file)
        image_path = image_file

        image_array, mask_img = self.preprocess_img(image_path, mask_path)

        return image_array, mask_img

    def transform_batch_images(self, batch_x):
        if self.augmentation:
            sometimes = lambda aug: iaa.Sometimes(0.25, aug)
            augmenter = iaa.Sequential(
                [
                    iaa.Fliplr(0.25),
                    iaa.Flipud(0.10),
                    iaa.geometric.Affine(
                        rotate=(-45, 45), order=1, mode="constant", fit_output=False
                    ),
                    sometimes(
                        iaa.Crop(px=(0, 25), keep_size=True, sample_independently=False)
                    ),
                ],
                random_order=True,
            )
            batch_x = augmenter.augment_images(batch_x)
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError(
                """
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """
            )
        return self.y_path[: self.steps * self.batch_size]

    def get_x_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError(
                """
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """
            )
        return self.x_path[: self.steps * self.batch_size]

    def prepare_dataset(self):
        # df = self.dataset_df.sample(frac=1.0, random_state=self.random_state)
        df = self.dataset_df
        self.x_path, self.y_path = (df[self.x_col].values, 
                                df[self.y_col].values)

        # self.y = np.where(self.y=='abnormal', 1, 0)

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

    def preprocess_img(self, path, mask_path):
        preprocessor = SingleImagePreprocessorSeg(target_size=self.target_size,
                                                add_channel=self.add_channel)
        img = preprocessor.read_img_mask(path, mask_path)
        preprocessed_img, mask_img = preprocessor.preprocess_image(img)
        return preprocessed_img, mask_img
    