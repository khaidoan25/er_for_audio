from train_model_rgb_1 import model
import numpy as np
import tensorflow as tf
import h5py
import os
from tqdm import tqdm
from keras import backend as K
from keras.metrics import categorical_crossentropy

if __name__ == "__main__":
    X_test_1 = h5py.File("dataset_RGB_1.h5", "r")["X_te"][:]
    y_test_1 = h5py.File("dataset_RGB_1.h5", "r")["y_te"][:]
    X_test_2 = h5py.File("dataset_RGB_2.h5", "r")["X_te"][:]
    y_test_2 = h5py.File("dataset_RGB_2.h5", "r")["y_te"][:]
    X_test_3 = h5py.File("dataset_RGB_3.h5", "r")["X_te"][:]
    y_test_3 = h5py.File("dataset_RGB_3.h5", "r")["y_te"][:]

    model_1 = model()
    model_1.load_weights("ckpt_cnn/CNN_1.h5")
    model_2 = model()
    model_2.load_weights("ckpt_cnn/CNN_2.h5")
    model_3 = model()
    model_3.load_weights("ckpt_cnn/CNN_3.h5")

    y_test_1 = tf.keras.utils.to_categorical(y_test_1, num_classes=6)
    y_test_2 = tf.keras.utils.to_categorical(y_test_2, num_classes=6)
    y_test_3 = tf.keras.utils.to_categorical(y_test_3, num_classes=6)

    _, eval_1 = model_1.evaluate(X_test_1, y_test_1)
    _, eval_2 = model_2.evaluate(X_test_2, y_test_2)
    _, eval_3 = model_3.evaluate(X_test_3, y_test_3)

    choose = np.argmax([eval_1, eval_2, eval_3])

    if choose == 0:
        model = model_1
    elif choose == 1:
        model = model_2
    else:
        model = model_3
    
    X_test = h5py.File("test_RGB.h5", "r")["X"][:]

    preds = model.predict(X_test)
    np.save("softmax_RGB.npy", preds)