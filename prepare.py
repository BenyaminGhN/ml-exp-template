from pathlib import Path
from omegaconf import OmegaConf

from src.data_ingestion import prepare_dir, make_classification_df, make_segmentation_df

def main():
    # get config file
    config_file_path = Path('./config.yaml')
    config = OmegaConf.load(config_file_path)

    data_root_dir = config.data_root_dir

    # prepare the data directory
    prepare_dir(data_root_dir)
    # create target csv file for classifcation task
    make_classification_df(config)
    # create target csv file for segmentation task
    make_segmentation_df(config)

if __name__ == '__main__':
    main()