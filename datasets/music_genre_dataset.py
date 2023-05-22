import torch
import os
from pytorch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio

class MusicGenreDataset(Dataset):
    def __init__(self, audio_dir, annotations_file, transforms=None):
        self.audio_dir = audio_dir
        self.annotations = pd.read_csv(annotations_file, usecols=["track_id", "genre_top"])
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_sample_path = self._get_audio_sample_path(idx)
        label = self._get_audio_sample_label(idx)
        signal, sr = torchaudio.load(audio_sample_path)
        if self.transforms: 
            signal = self.transforms(signal)
        
        return signal, label
    
    def _get_audio_sample_path(self, idx):
        track_id = "track_id" + str(self.annotations.iloc[idx, 0])
        fileName = str(track_id) + ".mp3"
        path = os.path.join(self.audio_dir, fileName)
        
        return path
    
    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx, 1]
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = os.path.expanduser("/Desktop/fma/fma_metadata/tracks.csv")
    AUDIO_DIR = os.path.expanduser("/Desktop/fma/fma_small")

    mgd = MusicGenreDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print("There are " + len(mgd) + " samples in the dataset.")

    signal, label = mgd[0]

    a = 1