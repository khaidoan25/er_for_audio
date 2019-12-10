import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import h5py

# from google_drive_downloader import GoogleDriveDownloader as gdd
# import matplotlib.pyplot as plt
# from tensorflow.contrib.eager.python import tfe
# from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def generate_data(image_paths, size=224):

    image_array = np.zeros((len(image_paths), size, size), dtype='float32')
    
    for idx, image_path in tqdm(enumerate(image_paths)):
        ### START CODE HERE
          
        # Đọc ảnh bằng thư viện Pillow và resize ảnh
        image = Image.open(image_path).convert('L')
        image = image.resize((size, size))
        # image = np.asarray( image, dtype="int32" )
        

        # Chuyển ảnh thành numpy array và gán lại mảng image_array
        temp = np.asarray( image, dtype="float16" )
        # temp = temp[...,np.newaxis]
        image_array[idx,:,:] = temp
        
        ### END CODE HERE
    return image_array

def trans(x):
    x = x[:-3]
    x = x + "jpg"
    return x
  
  # List các đường dẫn file cho việc huấn luyện

'''def processing():

    train_files = [os.path.join("mel_CNN/", trans(file)) for file in train_df.File]

    # List các nhãn
    train_y = train_df.Label

    # Tạo numpy array cho dữ liệu huấn luyện
    # train_arr = generate_data(train_files)
    train_arr = load('train_arr.npy')

    print(train_arr.shape)

    num_classes = len(np.unique(train_y))
    y_ohe = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)
    return train_arr, y_ohe
'''

if __name__ == "__main__":
    train_file = pd.read_csv("train_label.csv")["File"]

    train_files = [os.path.join('mel_CNN_train/', trans(file)) for file in train_file]
    test_files = [os.path.join('mel_CNN_test/', trans(file)) for file in glob(os.path.join("Dataset/TestSet", "*"))]

    train = generate_data(train_files)
    test = generate_data(test_files)

    with h5py.File("dataset_gray.h5", "w") as hf:
        hf.create_dataset("X_tr", data=train)
        hf.create_dataset("y_tr", data=np.array(train_file["Label"]))
        hf.create_dataset("X_te", data=test)