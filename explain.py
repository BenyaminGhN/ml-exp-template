from pathlib import Path
from pydoc import locate
from omegaconf import OmegaConf
import tensorflow as tf
tfk = tf.keras
K = tfk.backend

from src.data_preparation import DataLoader
from src.interpertation import Explainer
from src.utils import get_checkpoints_info

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

    explainer = Explainer(config=config,
                          model=model)

    explainer.explain(eval_seq)
    

if __name__ == "__main__":
    main()