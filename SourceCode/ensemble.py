from tqdm import tqdm
from glob import glob
import h5py
import os
import json
import numpy as np
import pandas as pd

if __name__ == "__main__":
    w_stft = np.array([0.43190648, 0.23611884, 0.31559783, 0.47548496, 0.50917835, 0.23090892])
    w_gray = np.array([0.35361511, 0.47548496, 0.25838965, 0.2883962,  0.3772086,  0.25519382])
    w_rgb = np.array([0.21447841, 0.2883962, 0.42601251, 0.23611884, 0.11361305, 0.51389725])
    # Thêm weight ở đây

    softmax_stft = np.load("softmax_stft.npy")
    softmax_gray = np.load("softmax_gray.npy")
    softmax_rgb = np.load("softmax_RGB.npy")
    # Thêm softmax ở đây

    y_pred = []
    for sm_stft, sm_gray, sm_rgb in zip(softmax_stft, softmax_gray, softmax_rgb):
        softmax = sm_stft*w_stft + sm_gray*w_gray + sm_rgb*w_rgb# Thêm softmax * weight ở đây
        pred = np.argmax(softmax)
        y_pred.append(pred)

    test_files = [file for file in glob(os.path.join("../Dataset/TestSet", "*"))]
    df = pd.DataFrame()
    df["File"] = test_files
    df["Label"] = y_pred
    df.to_csv("submission.csv", index=False)