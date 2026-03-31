
import librosa
import numpy as np
import os

def extract_and_save_mfcc(wav_path, output_path):
    y, sr = librosa.load(wav_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    np.save(output_path, mfcc)
    return mfcc
