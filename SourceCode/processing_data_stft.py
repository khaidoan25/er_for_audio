import librosa
import numpy as np
from glob import glob
import os
import pandas as pd
import h5py
from tqdm import tqdm


if __name__ == "__main__":
    
    train_1 = pd.read_csv("train_1.csv")
    test_1 = pd.read_csv("test_1.csv")
    train1 = []
    test1 = []
    # test_files = [file for file in glob(os.path.join("../Dataset/TestSet", "*"))]

    for file_name in tqdm(train_1.File):
        raw_signal, sample_rate = librosa.load(os.path.join("../Dataset/TrainSet", file_name))
        # TRIM SILENT EDGES
        trimmed_signal, _ = librosa.effects.trim(raw_signal)
        if len(trimmed_signal) < 60000:
            pad = np.zeros(60000-len(trimmed_signal))
            final_signal = np.append(trimmed_signal, pad)
        else:
            final_signal = trimmed_signal[:60000]
        stft = np.abs(librosa.core.stft(final_signal, n_fft=512, hop_length=512)).transpose()
        train1.append(stft)

    for file_name in tqdm(test_1.File):
        raw_signal, sample_rate = librosa.load(os.path.join("../Dataset/TrainSet", file_name))
        # TRIM SILENT EDGES
        trimmed_signal, _ = librosa.effects.trim(raw_signal)
        if len(trimmed_signal) < 60000:
            pad = np.zeros(60000-len(trimmed_signal))
            final_signal = np.append(trimmed_signal, pad)
        else:
            final_signal = trimmed_signal[:60000]
        stft = np.abs(librosa.core.stft(final_signal, n_fft=512, hop_length=512)).transpose()
        test1.append(stft)

    with h5py.File("dataset_stft_1.h5", "w") as hf:
        hf.create_dataset("X_tr", data=np.array(train1))
        hf.create_dataset("y_tr", data=np.array(train_1["Label"]))
        hf.create_dataset("X_te", data=np.array(test1))
        hf.create_dataset("y_te", data=np.array(test_1.Label))

    train_2 = pd.read_csv("train_2.csv")
    test_2 = pd.read_csv("test_2.csv")
    train2 = []
    test2 = []
    # test_files = [file for file in glob(os.path.join("../Dataset/TestSet", "*"))]

    for file_name in tqdm(train_2.File):
        raw_signal, sample_rate = librosa.load(os.path.join("../Dataset/TrainSet", file_name))
        # TRIM SILENT EDGES
        trimmed_signal, _ = librosa.effects.trim(raw_signal)
        if len(trimmed_signal) < 60000:
            pad = np.zeros(60000-len(trimmed_signal))
            final_signal = np.append(trimmed_signal, pad)
        else:
            final_signal = trimmed_signal[:60000]
        stft = np.abs(librosa.core.stft(final_signal, n_fft=512, hop_length=512)).transpose()
        train2.append(stft)

    for file_name in tqdm(test_2.File):
        raw_signal, sample_rate = librosa.load(os.path.join("../Dataset/TrainSet", file_name))
        # TRIM SILENT EDGES
        trimmed_signal, _ = librosa.effects.trim(raw_signal)
        if len(trimmed_signal) < 60000:
            pad = np.zeros(60000-len(trimmed_signal))
            final_signal = np.append(trimmed_signal, pad)
        else:
            final_signal = trimmed_signal[:60000]
        stft = np.abs(librosa.core.stft(final_signal, n_fft=512, hop_length=512)).transpose()
        test2.append(stft)

    with h5py.File("dataset_stft_2.h5", "w") as hf:
        hf.create_dataset("X_tr", data=np.array(train2))
        hf.create_dataset("y_tr", data=np.array(train_2["Label"]))
        hf.create_dataset("X_te", data=np.array(test2))
        hf.create_dataset("y_te", data=np.array(test_2.Label))

    train_3 = pd.read_csv("train_3.csv")
    test_3 = pd.read_csv("test_3.csv")
    train3 = []
    test3 = []
    # test_files = [file for file in glob(os.path.join("../Dataset/TestSet", "*"))]

    for file_name in tqdm(train_3.File):
        raw_signal, sample_rate = librosa.load(os.path.join("../Dataset/TrainSet", file_name))
        # TRIM SILENT EDGES
        trimmed_signal, _ = librosa.effects.trim(raw_signal)
        if len(trimmed_signal) < 60000:
            pad = np.zeros(60000-len(trimmed_signal))
            final_signal = np.append(trimmed_signal, pad)
        else:
            final_signal = trimmed_signal[:60000]
        stft = np.abs(librosa.core.stft(final_signal, n_fft=512, hop_length=512)).transpose()
        train3.append(stft)

    for file_name in tqdm(test_3.File):
        raw_signal, sample_rate = librosa.load(os.path.join("../Dataset/TrainSet", file_name))
        # TRIM SILENT EDGES
        trimmed_signal, _ = librosa.effects.trim(raw_signal)
        if len(trimmed_signal) < 60000:
            pad = np.zeros(60000-len(trimmed_signal))
            final_signal = np.append(trimmed_signal, pad)
        else:
            final_signal = trimmed_signal[:60000]
        stft = np.abs(librosa.core.stft(final_signal, n_fft=512, hop_length=512)).transpose()
        test3.append(stft)

    with h5py.File("dataset_stft_3.h5", "w") as hf:
        hf.create_dataset("X_tr", data=np.array(train3))
        hf.create_dataset("y_tr", data=np.array(train_3["Label"]))
        hf.create_dataset("X_te", data=np.array(test3))
        hf.create_dataset("y_te", data=np.array(test_3.Label))


    test_files = [file for file in glob(os.path.join("../Dataset/TestSet", "*"))]
    test = []
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

    with h5py.File("test_stft.h5", "w") as hf:
        hf.create_dataset("X", data=np.array(test))