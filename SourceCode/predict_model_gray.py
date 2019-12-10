from train_model_gray import model_gray
import numpy as np
import tensorflow as tf
import h5py
import os
from tqdm import tqdm

if __name__ == "__main__":
    
    model = model_gray()
    model.load_weights("ckpt_gray/model_gray.h5")

    with h5py.File(os.path.join("dataset_gray.h5"), "r") as hf:
        # X_train = np.expand_dims(hf["X_tr"][:], axis=-1)
        # y_train = hf["y_tr"][:]
        X_test = np.expand_dims(hf["X_te"][:], axis=-1)

    softmax_pred = []

    for x in tqdm(X_test):
        softmax = np.squeeze(model.predict(tf.expand_dims(np.asarray(x, dtype=np.float16), axis=0)))
        softmax_pred.append(softmax)

    np.save("softmax_gray.npy", np.array(softmax))