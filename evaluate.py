import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pydoc import locate
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers

from src.data_preparation import DataLoader
from src.utils import f1, get_checkpoints_info


def main():

    # get config file
    config_file_path = Path('config.yaml')
    config = OmegaConf.load(config_file_path)

    data_root_dir = config.data_root_dir

    # create data generators
    data_loader = DataLoader(config=config, 
                             data_dir=data_root_dir)
    
    eval_seq, _ = data_loader.create_eval_generator()

    # get the best checkpoint
    checkpoints = get_checkpoints_info(Path(config.run_dir).joinpath('checkpoints'))
    if config.training.export_mode == 'min':
        selected_checkpoint = min(checkpoints, key=lambda x: x['value'])
    else:
        selected_checkpoint = max(checkpoints, key=lambda x: x['value'])

    # get model architecture
    model = locate(config.model)(config=config).get_compiled_model()

    # load the best saved model
    model.load_weights(selected_checkpoint['path'])

    # get predictions
    predictions = model.predict(eval_seq)

    if config.data_pipeline.is_seg:
        predictions = []
        prediction_maps = predictions
        for i, mask in enumerate(prediction_maps):
            decision_tresh = config.model_builder.threshold
            predictions.append(int(bool(np.sum(np.where(mask>=decision_tresh, 1, 0)))))
    else:
        if predictions.shape[1]==1:
            predictions = predictions.reshape((-1, ))
        else:
            predictions = predictions[:, 1].reshape((-1, ))

    # save the predictions
    eval_df = eval_seq.get_dataset_df()
    siuid_list = eval_df['SOPInstanceUID'].values
    dataset_df = pd.DataFrame([siuid_list, predictions], index=['SOPInstanceUID', 'Label']).T
    tresh = config.model_builder.threshold
    dataset_df['Preds'] = predictions
    dataset_df['Label'] = np.where(predictions>=tresh, 1, 0).astype('int')
    prediction_df = dataset_df[['SOPInstanceUID', 'Label', 'Preds']]
    prediction_df.to_csv(os.path.join(data_root_dir, config.results_csv), index=False)
    

if __name__ == '__main__':
    main()