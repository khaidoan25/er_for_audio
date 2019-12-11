import librosa
import numpy as np
from glob import glob
import os
import pandas as pd
import h5py
from tqdm import tqdm


if __name__ == "__main__":
    
    train_file = pd.read_csv("train_label.csv")
    train = []
    test = []
    test_files = [file for file in glob(os.path.join("../Dataset/TestSet", "*"))]

    for file_name in tqdm(train_file.File):
        raw_signal, sample_rate = librosa.load(os.path.join("../Dataset/TrainSet", file_name))
        # TRIM SILENT EDGES
        trimmed_signal, _ = librosa.effects.trim(raw_signal)
        if len(trimmed_signal) < 60000:
            pad = np.zeros(60000-len(trimmed_signal))
            final_signal = np.append(trimmed_signal, pad)
        else:
            final_signal = trimmed_signal[:60000]
        stft = np.abs(librosa.core.stft(final_signal, n_fft=512, hop_length=512)).transpose()
        train.append(stft)

    for file_name in tqdm(test_files):
        raw_signal, sample_rate = librosa.load(file_name)
        # TRIM SILENT EDGES
        trimmed_signal, _ = librosa.effects.trim(raw_signal)
        if len(trimmed_signal) < 60000:
            pad = np.zeros(60000-len(trimmed_signal))
            final_signal = np.append(trimmed_signal, pad)
        else:
            final_signal = trimmed_signal[:60000]
        stft = np.abs(librosa.core.stft(final_signal, n_fft=512, hop_length=512)).transpose()
        test.append(stft)

    with h5py.File("dataset_stft.h5", "w") as hf:
        hf.create_dataset("X_tr", data=np.array(train))
        hf.create_dataset("y_tr", data=np.array(train_file["Label"]))
        hf.create_dataset("X_te", data=np.array(test))