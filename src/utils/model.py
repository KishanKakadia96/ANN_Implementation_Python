import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os
def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS, NUM_CLASSES):
    LAYERS= [tf.keras.layers.Flatten(input_shape=[28,28], name="inputlayer"),
    tf.keras.layers.Dense(300, activation="relu", name="hiddenlayer1"),
    tf.keras.layers.Dense(100, activation="relu", name="hiddenlayer2"),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputlayer1")]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    # Already initialized in config.yaml file
    # LOSS_FUNCTION = "sparse_categorical_crossentropy"  # use => tf.losses.sparse_categorical_crossentropy
    # OPTIMIZER = "SGD"  # or use with custom learning rate => tf.keras.optimizer.SGD(0.02)
    # METRICS = ["accuracy"]

    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)


    return model_clf  ##<<< untrained model

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m_%d_%H%M%S_{filename}")
    return unique_filename
def save_model(model,model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model= os.path.join(model_dir,unique_filename)
    model.save(path_to_model)

def save_plots(df, plot_name, plot_dir):
    unique_filename = get_unique_filename(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_filename)
    df.plot(figsize=(10, 7))
    plt.grid(True)
    plt.savefig(path_to_plot)
    plt.show()

