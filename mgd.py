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
        
        try:
            # Convert mp3 to wav using PyDub
            audio = AudioSegment.from_mp3(audio_sample_path)
            wav_path = audio_sample_path.replace('.mp3', '.wav')
            audio.export(wav_path, format="wav")
            
            signal, sr = torchaudio.load(wav_path)
          
            signal = self.resample(signal)

            if (self.apply_spectrogram):
                signal = self.spectro(signal)

            return signal, label, audio_sample_path
        
        except: 
            pass

    
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