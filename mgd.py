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


class MusicGenreDataset(Dataset):
    def __init__(self, audio_dir, annotations_file, apply_spectrogram=False):
        self.audio_dir = audio_dir
        self.annotations = annotations_file
        self.apply_spectrogram = apply_spectrogram

        self.resample = T.Resample(orig_freq=44100, new_freq=8000) # reduce frequency
        self.spectro = T.Spectrogram(n_fft=400, win_length=400) # convert audio signal into spectrogram
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_sample_path = self._get_audio_sample_path(idx)
        label = self._get_audio_sample_label(idx)
        
        # Convert mp3 to wav using PyDub
        audio = AudioSegment.from_mp3(audio_sample_path)
        wav_path = audio_sample_path.replace('.mp3', '.wav')
        audio.export(wav_path, format="wav")
        
        signal, sr = torchaudio.load(wav_path)
        try:  
            signal = self.resample(signal)

            if (self.apply_spectrogram):
                signal = self.spectro(signal)

            return signal, label
        
        except AttributeError: 
            pass

    
    def _get_audio_sample_path(self, idx):
        poss_folder = [str(i).zfill(3) for i in range(156)] # possible folders where the audio samples could be located
        
        track_id = str(self.annotations.iloc[idx, 0]).zfill(3) # take the num of the file in track_id coloum and add zeros if necessary
        
        for folder in poss_folder: 
            filename = f"{folder}{track_id}.mp3" # concatenate the folder name with the track ID and .mp3
            path = os.path.join(self.audio_dir, folder, filename)
                
            if os.path.exists(path):
                return path
            
        return None
 
    
    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx, 1]