import os
import torch 
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import fnmatch
from pydub import AudioSegment
import torchaudio.transforms as T
import glob
import torch.nn as nn

import wave
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import librosa



class MusicGenreDataset(Dataset):
    def __init__(self, audio_dir, annotations_file):
        self.audio_dir = audio_dir
        self.annotations = annotations_file

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_sample_path = self._get_audio_sample_path(idx)
        label = self._get_audio_sample_label(idx)
        
        try:
            # Convert mp3 to wav using PyDub
            audio = AudioSegment.from_mp3(audio_sample_path)
            wav_path = audio_sample_path.replace('.mp3', '.wav')
            audio.export(wav_path, format="wav")
            signal, sr = librosa.load(wav_path, sr=None) # load audio 
            D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max) # create spectrogram
        

            # make sure the spectrogram size is divisible by 128
            height, width = D.shape
            D = D[:height - height % 128, :width - width % 128]

            slices = []

            # slice the spectrogram
            for i in range(0, D.shape[0], 128):
                for j in range(0, D.shape[1], 128):
                    slice = D[i:i+128, j:j+128]

                    # convert the slice into a PyTorch tensor
                    tensor = torch.from_numpy(slice)
                    slices.append(tensor)

            return D, slices, label

        except Exception as e:
            print("An error occurred: ", e)
    
    def _get_audio_sample_path(self, idx):
        track_id = str(self.annotations.iloc[idx, 0]).zfill(3)

        if len(track_id)>3: 
            file = track_id.zfill(6)
            filename = f"{file}.mp3"
            folder = filename[:3]
            path = os.path.join(self.audio_dir, folder, filename)
            return path

        elif len(track_id)==3: 
            poss_folder = a = [str(i).zfill(3) for i in range(156)]

            for folder in poss_folder: 
                filename = f"{folder}{track_id}.mp3"
                path = os.path.join(self.audio_dir, folder, filename)
                if os.path.exists(path):
                    return path   
        
            
        raise FileNotFoundError(f"No file found for index {idx} with track_id {track_id} path {path}")
 
    
    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx, 1]