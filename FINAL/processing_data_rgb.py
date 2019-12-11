import os
import numpy as np
np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from numpy import load
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def generate_data(image_paths, size=224):

    image_array = np.zeros((len(image_paths), size, size,3), dtype='float32')
    
    for idx, image_path in tqdm(enumerate(image_paths)):
        image = Image.open(image_path)
        image = image.resize((size, size))
        temp = np.asarray( image, dtype="float32" )
        temp /= 255
        image_array[idx,:,:,:] = temp
    return image_array


def trans(x):
  x = x[:-3]
  x = x + "jpg"
  return x

if __name__ == "__main__":
    train_file = pd.read_csv("train_label.csv")["File"]

    train_files = [os.path.join('mel_CNN_train/', trans(file)) for file in train_file]
    test_files = [os.path.join('mel_CNN_test/', trans(file)) for file in glob(os.path.join("Dataset/TestSet", "*"))]

    train = generate_data(train_files)
    test = generate_data(test_files)

    with h5py.File("dataset_RGB.h5", "w") as hf:
        hf.create_dataset("X_tr", data=train)
        hf.create_dataset("y_tr", data=np.array(train_file["Label"]))
        hf.create_dataset("X_te", data=test)