from train_model_stft_1 import model_stft
import numpy as np
import tensorflow as tf
import h5py
import os
from tqdm import tqdm


if __name__ == "__main__":
    
    X_test_1 = h5py.File("dataset_stft_1.h5", "r")["X_te"][:]
    y_test_1 = h5py.File("dataset_stft_1.h5", "r")["y_te"][:]
    X_test_2 = h5py.File("dataset_stft_2.h5", "r")["X_te"][:]
    y_test_2 = h5py.File("dataset_stft_2.h5", "r")["y_te"][:]
    X_test_3 = h5py.File("dataset_stft_3.h5", "r")["X_te"][:]
    y_test_3 = h5py.File("dataset_stft_3.h5", "r")["y_te"][:]

    model_1 = model_stft(118, 257)
    model_1.load_weights("ckpt_stft/model_stft_1.h5")
    model_2 = model_stft(118, 257)
    model_2.load_weights("ckpt_stft/model_stft_2.h5")
    model_3 = model_stft(118, 257)
    model_3.load_weights("ckpt_stft/model_stft_3.h5")

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
    
    X_test = h5py.File("test_stft.h5", "r")["X"][:]

    softmax_pred = []

    for x in tqdm(X_test):
        softmax = np.squeeze(model.predict(tf.expand_dims(x, axis=0)))
        softmax_pred.append(softmax)

    np.save("softmax_stft.npy", np.array(softmax_pred))