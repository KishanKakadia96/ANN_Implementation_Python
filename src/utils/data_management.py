import tensorflow as tf
import time
import os


def get_data(validation_datasize):
    mnist= tf.keras.datasets.mnist
    
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
   
    #creating a validation datasets from the full training data
    #Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    X_valid, X_train = X_train_full[:validation_datasize]/255., X_train_full[validation_datasize:]/255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    #scale the test set as well  
    X_test = X_test / 255.
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

#creating tensorboard_logs directory with uniqueName
def get_log_path(log_dir="logs/fit"):
  uniqueName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
  log_path = os.path.join(log_dir, uniqueName)
  print(f"savings logs at: {log_path}")

  return log_path

