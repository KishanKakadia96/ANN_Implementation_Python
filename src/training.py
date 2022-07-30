from gc import callbacks
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from src.utils.common import read_config  
from src.utils.data_management import get_data
from src.utils.model import create_model,save_model, save_plots
from src.utils.callbacks import get_callbacks
import argparse

# os.chdir('C:\\Kishan\\Github\\Artificial_Neural_Network\\ANN_Implementation_Python')

def training(config_path):
    config = read_config(config_path)

    validation_datasize = config["params"]["validation_datasize"]
    
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)


    LOSS_FUNCTION = config["params"]["loss_functions"]
    OPTIMIZER = config["params"]["optimizer"]   
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION,OPTIMIZER,METRICS, NUM_CLASSES)

    CALLBACKS_LIST = get_callbacks(config, X_train)

    EPOCHS = config["params"]["epochs"] 
    VALIDATION = (X_valid, y_valid)

    history= model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, callbacks=CALLBACKS_LIST)
    #Save model function

    artifacts_dir= config["artifacts"]["artifacts_dir"]
    model_dir= config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["model_name"]
    save_model(model,model_name,model_dir_path)

    #Save plots

    plot_dir = config["artifacts"]["plot_dir"]
    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)#
    plot_name = config["artifacts"]["plot_name"]

    df=pd.DataFrame(history.history)
    save_plots(df, plot_name, plot_dir_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c",default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)