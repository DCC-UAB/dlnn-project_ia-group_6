from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def plot_spectrogram(spectrogram):
    plt.figure(figsize=(10, 5))
    
    spectrogram_np = spectrogram.numpy()
    
    spectrogram_np = np.mean(spectrogram_np, axis=0)
    spectrogram_np = np.log(spectrogram_np + 1e-9)  #  log scale to make visualization easier
    plt.imshow(spectrogram_np, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Frames')
    plt.ylabel('Frequency bin')
    plt.tight_layout()
    plt.show()


def splits_n_dataloaders(data): 
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)
    
    batch_size = 32

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader

def split_X_y(dtset): 
    X = []
    y = []
    
    try: 
        for s,l,p in dtset: 
            newtensor=s
            X.append(newtensor)
            y.append(l)
    
    except Exception as e : 
        return f"There was an error: {e}"
    
    return X, y
    

def filter_small_subset(df):
    return df[df['subset'] == 'small']


def load(filepath):
    
    filename = os.path.basename(filepath)
    file = pd.read_csv(filename)

    small_subset = filter_small_subset(file)

    interest = small_subset[['track_id', 'genre_top']]

    return interest