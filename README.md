[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/wT71nrpQ)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11110461&assignment_repo_type=AssignmentRepo)

# Music Genre Classification
Our project aimed to conduct music analysis. We were provided with the archive fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip, which consisted of 8000 songs of 30s in mp3 format. Accompanied by a csv file containing relevant track information such as genre, album, listeners and artist, that can be found in TRACKS.csv.zip file. 

## Code structure
Our code is taking the raw mp3 files from the fma_small directory, and the reduced tracks.csv(small). Then it goes through the music_genre_dataset class that can be found in the mgd.py. It converts the mp3 to wav (the format that allows spectrogram conversion), then creates a spectrogram for each of these tracks, and then it splits in parts of 128x128, and finally converts them into tensors. These returns a dataset containing the slices and the label for that group of slices representing each of the tracks. Later we separate all the slices and assign them the corresponding label, and we end up with two tensor lists X and y cotaining more than 714,000 files. We split all these into train, test and validation sets and created the dataloaders, so we could pass the files through the convolutional neural network that can be found in model.py. 

During the training process we have had trouble since our computers were not able to handle such amount of data. For this reason we did an experiment with a smaller amount of tracks (60), so we could see how the model works. In this case we obtained a maximum accuracy of 60%.  

## Required files 
fma_small

TRACKS.csv

main.ipynb

mgd.py

s.py

utils -> utils.py

models -> model.py



## Contributors
__Nora Mendoza:__ nora.mendoza@autonoma.cat

__Roger Perramon:__ roger.perramon@autonoma.cat

__Marc Garreta:__ marc.garreta@autonoma.cat


Xarxes Neuronals i Aprenentatge Profund
Grau de __Artificial Intelligence__, 
UAB, 2023
