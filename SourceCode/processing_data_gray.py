import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import h5py
from PIL import Image

def generate_data(image_paths, size=224):

    image_array = np.zeros((len(image_paths), size, size), dtype='float32')
    
    for idx, image_path in tqdm(enumerate(image_paths)):
        image = Image.open(image_path).convert('L')
        image = image.resize((size, size))
        temp = np.asarray( image, dtype="float16")
        image_array[idx,:,:] = temp
        
    return image_array

def trans(x):
    x = x[:-3]
    x = x + "jpg"
    return x

if __name__ == "__main__":
    train_1 = pd.read_csv("train_1.csv")
    test_1 = pd.read_csv("test_1.csv")

    train_1s = [os.path.join('mel_CNN_train/', trans(file)) for file in train_1.File]
    # test_files = [os.path.join('mel_CNN_test/', trans(file)) for file in glob(os.path.join("../Dataset/TestSet", "*"))]
    test_1s = [os.path.join('mel_CNN_train/', trans(file)) for file in test_1.File]

    train1 = generate_data(train_1s)
    test1 = generate_data(test_1s)

    with h5py.File("dataset_gray_1.h5", "w") as hf:
        hf.create_dataset("X_tr", data=train1)
        hf.create_dataset("y_tr", data=np.array(train_1["Label"]))
        hf.create_dataset("X_te", data=test1)
        hf.create_dataset("y_te", data=np.array(test_1.Label))

    train_2 = pd.read_csv("train_2.csv")
    test_2 = pd.read_csv("test_2.csv")

    train_2s = [os.path.join('mel_CNN_train/', trans(file)) for file in train_2.File]
    # test_files = [os.path.join('mel_CNN_test/', trans(file)) for file in glob(os.path.join("../Dataset/TestSet", "*"))]
    test_2s = [os.path.join('mel_CNN_train/', trans(file)) for file in test_2.File]

    train2 = generate_data(train_2s)
    test2 = generate_data(test_2s)

    with h5py.File("dataset_gray_2.h5", "w") as hf:
        hf.create_dataset("X_tr", data=train2)
        hf.create_dataset("y_tr", data=np.array(train_2["Label"]))
        hf.create_dataset("X_te", data=test2)
        hf.create_dataset("y_te", data=np.array(test_2.Label))

    train_3 = pd.read_csv("train_3.csv")
    test_3 = pd.read_csv("test_3.csv")

    train_3s = [os.path.join('mel_CNN_train/', trans(file)) for file in train_3.File]
    # test_files = [os.path.join('mel_CNN_test/', trans(file)) for file in glob(os.path.join("../Dataset/TestSet", "*"))]
    test_3s = [os.path.join('mel_CNN_train/', trans(file)) for file in test_3.File]

    train3 = generate_data(train_3s)
    test3 = generate_data(test_3s)

    with h5py.File("dataset_gray_3.h5", "w") as hf:
        hf.create_dataset("X_tr", data=train3)
        hf.create_dataset("y_tr", data=np.array(train_3["Label"]))
        hf.create_dataset("X_te", data=test3)
        hf.create_dataset("y_te", data=np.array(test_3.Label))

    test_files = [os.path.join('mel_CNN_test/', trans(file)) for file in glob(os.path.join("../Dataset/TestSet", "*"))]
    test = generate_data(test_files)
    with h5py.File("test_gray.h5", "w") as hf:
        hf.create_dataset("X", data=test)