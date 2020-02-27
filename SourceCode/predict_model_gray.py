from train_model_gray_1 import model_gray
import numpy as np
import tensorflow as tf
import h5py
import os
from tqdm import tqdm

if __name__ == "__main__":
    
    X_test_1 = np.expand_dims(h5py.File("dataset_gray_1.h5", "r")["X_te"][:], axis=-1)
    y_test_1 = h5py.File("dataset_gray_1.h5", "r")["y_te"][:]
    X_test_2 = np.expand_dims(h5py.File("dataset_gray_2.h5", "r")["X_te"][:], axis=-1)
    y_test_2 = h5py.File("dataset_gray_2.h5", "r")["y_te"][:]
    X_test_3 = np.expand_dims(h5py.File("dataset_gray_3.h5", "r")["X_te"][:], axis=-1)
    y_test_3 = h5py.File("dataset_gray_3.h5", "r")["y_te"][:]

    model_1 = model_gray()
    model_1.load_weights("ckpt_gray/model_gray_1.h5")
    model_2 = model_gray()
    model_2.load_weights("ckpt_gray/model_gray_2.h5")
    model_3 = model_gray()
    model_3.load_weights("ckpt_gray/model_gray_3.h5")

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
    
    X_test = np.expand_dims(h5py.File("test_gray.h5", "r")["X"][:], axis=-1)

    softmax_pred = []

    for x in tqdm(X_test):
        softmax = np.squeeze(model.predict(tf.expand_dims(np.asarray(x, dtype=np.float16), axis=0)))
        softmax_pred.append(softmax)

    np.save("softmax_gray.npy", np.array(softmax_pred))