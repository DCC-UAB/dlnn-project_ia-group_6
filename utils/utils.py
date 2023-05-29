from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


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
    train_size = int(0.8 * len(data))
    val_size = (len(data) - train_size) // 2
    test_size = len(data) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

    batch_size = 32

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader

