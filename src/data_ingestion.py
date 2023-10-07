import numpy as np
import pandas as pd
import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import pydicom as dicom

RANDOM_STATE = 2

def prepare_dir(data_root_dir='./data/'):
    """prepare the data directory for desired data folders and versions
    Descriptions:
    - these configs are adjusted for the iaaa dental detection competetion.
    """
    # version 1
    data_dir = 'content/iaaa-data/'
    sr_dir = os.path.join(data_root_dir, data_dir)
    if os.path.exists(sr_dir):
        ds_dir = os.path.join(data_root_dir, 'iaaa_data_v2/')
        shutil.move(sr_dir, ds_dir);
        # shutil.rmtree(sr_dir);

    #version 2
    data_dir = 'iaaa-data-v3/'
    sr_dir = os.path.join(data_root_dir, data_dir)
    if os.path.exists(sr_dir):
        ds_dir = os.path.join(data_root_dir, 'iaaa_data_v3/')
        shutil.move(sr_dir, ds_dir);
        # shutil.rmtree(sr_dir);

    #version 3
    data_dir = 'data-v3/'
    sr_dir = os.path.join(data_root_dir, data_dir)
    if os.path.exists(sr_dir):
        sr_dir_labeler = os.path.join(sr_dir, 'iaaa_dental_labeler6_dup/')
        if os.path.exists(sr_dir_labeler):
            ds_dir = os.path.join(data_root_dir, 'iaaa_data_v6/')
            shutil.move(sr_dir_labeler, ds_dir);
        
        sr_dir_labeler = os.path.join(sr_dir, 'iaaa_dental_labeler7_dup')
        if os.path.exists(sr_dir_labeler):
            ds_dir = os.path.join(data_root_dir, 'iaaa_data_v7/')
            shutil.move(sr_dir_labeler, ds_dir);

        # shutil.rmtree(sr_dir);

def make_classification_df(data_root_dir='./data/'):
    """prepare a structured csv file out of your data folder
        for classification task based on the instructions:

    Instructions:
        CSV file columns: 
            columns:
            - ID: the ids should be unique for each instance/row
            - Label: the label for the instance
            - Path: path of the data folder

    Descriptions:
    - these configs are adjusted for the iaaa dental detection competetion.
    - the image dataset are considered as dicom images (.dcm) which we name the ID column
        as SOPInstanceUID
    """
    data_dirs = [fn for fn in os.listdir(data_root_dir) 
                 if ('iaaa_data_v' in fn) and (os.path.isdir(os.path.join(data_root_dir, fn)))]

    dfs_list = []
    for data_dir in data_dirs:
        df_path = os.path.join(data_root_dir, data_dir, 'labels.csv')
        df = pd.read_csv(df_path)
        df['Path'] = df['SOPInstanceUID'].apply(lambda x: os.path.join(data_dir, 'DICOM', x+'.dcm'))
        dfs_list.append(df)

    columns = ['SOPInstanceUID', 'Label', 'Path']
    df_total = pd.concat(dfs_list, ignore_index =True).loc[:, columns]

    df_total.to_csv(os.path.join(data_root_dir, 'labels_total.csv'), index=False)

def make_segmentation_df(data_root_dir='./data/'):
    """prepare a structured csv file out of your data folder
        for segmentation task based on the instructions:

    Instructions:
        CSV file columns: 
            columns:
            - ID: the ids should be unique for each instance/row
            - Label: the label for the instance
            - Path: path of the data folder
            - MaskPath: path of the mask files
    Descriptions:
    - these configs are adjusted for the iaaa dental detection competetion.
    - the image dataset are considered as dicom images (.dcm) which we name the ID column
        as SOPInstanceUID
    """
    data_dirs = [fn for fn in os.listdir(data_root_dir) 
                 if ('iaaa_data' in fn) and (os.path.isdir(os.path.join(data_root_dir, fn)))]
    
    dfs_list = []
    for data_dir in data_dirs:
        df_path = os.path.join(data_root_dir, data_dir, 'labels.csv')
        df = pd.read_csv(df_path)
        df['Path'] = df['SOPInstanceUID'].apply(lambda x: os.path.join(data_dir, 'DICOM', x+'.dcm'))
        df['MaskPath'] = df['SOPInstanceUID'].apply(lambda x: os.path.join(data_dir, 'masks', x+'.png'))
        df['MaskPath_nii'] = df['SOPInstanceUID'].apply(lambda x: os.path.join(data_dir, 'masks', x+'.nii.gz'))
        dfs_list.append(df)

    columns = ['SOPInstanceUID', 'Label', 'Path', 'MaskPath', 'MaskPath_nii']
    df_total = pd.concat(dfs_list, ignore_index =True).loc[:, columns]

    df_total.to_csv(os.path.join(data_root_dir, 'labels_seg_total.csv'), index=False)


def get_class_dist(data_root_dir):
    """get the labels distributions
    """
    df_path = os.path.join(data_root_dir, 'labels.csv')
    df = pd.read_csv(df_path)

    labels = df['Label'].values
    class_names, class_dist = np.unique(labels, return_counts=True)
    print(class_names)

    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    bars = ax.bar(class_names, class_dist)

    plt.show()

def main():
    # get config file
    config_file_path = Path('./config.yaml')
    config = OmegaConf.load(config_file_path)

    data_root_dir = config.data_root_dir

    prepare_dir(data_root_dir)
    make_classification_df(data_root_dir)
    make_segmentation_df(data_root_dir)

if __name__ == '__main__':
    main()