# ml-exp-template
This repo is a sample template for ml/dl experimentation (vision) projects.

`note: The configs of this project are set for a vision classification task of opg dental images in terms of whether they need surgery or not.`

# Project Structure
the structure of the template is based on a `config` which is written in `.yaml` format.
the file is used to share paramters and settings of the project.

the `src` folder contains the function and classes needed to run the following operations:
- `prepare.py`: prepare data folder for training
- `train.py`: train the model on the prepared data 
- `evaluate.py`: evaluate the trained model
- `explain.py`: interpret the model

# How to Use
1. define your `config.yaml` file
2. `python prepare.py`: to create data the csv files needed for training and prepare data
3. `python train.py`: to train the model set in the config file and defined in the `model_building.py`
4. `python evaluate.py`: to evaluate your model and save the results
5. `python explain.py`: to explain your model with grad-cam or other XAI methods

