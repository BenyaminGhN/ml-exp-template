import os
import click
from pydoc import locate
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

from src.data_preparation import DataLoader
from src.utils import create_callbacks

np.random.seed(2)

def main():
    # get config file
    config_file_path = Path('config.yaml')
    config = OmegaConf.load(config_file_path)

    data_root_dir = config.data_root_dir

    # create data generators
    data_loader = DataLoader(config=config, 
                             data_dir=data_root_dir)
    
    train_seq, n_iter_train, val_seq, n_iter_val = data_loader.create_train_val_generator()
    class_weights = data_loader.get_class_weights()
    
    # define model for training
    model = locate(config.model)(config=config).get_compiled_model(class_weights)
    
    epochs = config.training.epochs
    callbacks = create_callbacks(config=config)
                                 
    history = model.fit(
        train_seq,
        steps_per_epoch=n_iter_train,
        epochs=epochs,
        validation_data=val_seq,
        validation_steps=n_iter_val,
        callbacks = callbacks
    )

if __name__ == "__main__":
    main()
