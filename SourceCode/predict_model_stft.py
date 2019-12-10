from train_model_stft import model_stft
import numpy as np
import tensorflow as tf
import h5py
import os
from tqdm import tqdm


if __name__ == "__main__":
    
    model = model_stft(118, 257)
    model.load_weights("ckpt_gray/model_stft.h5")

    with h5py.File(os.path.join("dataset_stft.h5"), "r") as hf:
        # X_train = np.expand_dims(hf["X_tr"][:], axis=-1)
        # y_train = hf["y_tr"][:]
        X_test = hf["X_te"][:]

    softmax_pred = []

    for x in tqdm(X_test):
        softmax = np.squeeze(model.predict(tf.expand_dims(x, axis=0)))
        softmax_pred.append(softmax)

    np.save("softmax_stft.npy", np.array(softmax))