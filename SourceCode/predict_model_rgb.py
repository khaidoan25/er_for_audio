from train_model_rgb import model
import numpy as np
import tensorflow as tf
import h5py
import os
from tqdm import tqdm

if __name__ == "__main__":
    model = model()
    model.load_weights("ckpt_cnn/CNN.h5")

    with h5py.File(os.path.join("dataset_RGB.h5"), "r") as hf:
        # X_train = np.expand_dims(hf["X_tr"][:], axis=-1)
        # y_train = hf["y_tr"][:]
        X_test = hf["X_te"][:]

    preds = model.predict(X_test)
    np.save("softmax_RGB.npy", preds)