import os
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd

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
        track_id = f"fold{self.annotations.iloc[idx, 0]}"
        fileName = f"{track_id}.mp3"
        path = os.path.join(self.audio_dir, fileName)
        
        return path
    
    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx, 1]
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = os.path.expanduser("~/Desktop/fma/fma_metadata/tracks.csv")
    AUDIO_DIR = os.path.expanduser("~/Desktop/fma/fma_small")

    mgd = MusicGenreDataset(AUDIO_DIR, ANNOTATIONS_FILE)

    print(f"There are {len(mgd)} samples in the dataset.")

    signal, label = mgd[0]