import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np
import tqdm
import json

def get_filename(srcfolder):
    filelist = os.listdir(srcfolder)
    return filelist

def create_spectogram(filename):
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    # spect = librosa.feature.delta(spect)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T

def convert_melspectogram_to_jpg(srcname, distname):
    sig, fs = librosa.load(srcname)
    save_path = distname

    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])

    ##S = librosa.feature.melspectrogram(y=sig, sr=fs)
    S = create_spectogram(srcname)
    librosa.display.specshow(S)
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()

if __name__ == "__main__":
    fileList_train = get_filename('Dataset/Train')

    for i in tqdm.tqdm(range(0, len(fileList_train))):
        srcname = 'Dataset/TrainSet/' + fileList_train[i]
        distname = 'mel_CNN_train/' + fileList_train[i].replace('.wav','.jpg')
        convert_melspectogram_to_jpg(srcname, distname)

    fileList_test = get_filename('Dataset/Test')

    for i in tqdm.tqdm(range(0, len(fileList_test))):
        srcname = 'Dataset/TestSet/' + fileList_test[i]
        distname = 'mel_CNN_test/' + fileList_test[i].replace('.wav','.jpg')
        convert_melspectogram_to_jpg(srcname, distname)