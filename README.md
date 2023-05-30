[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/wT71nrpQ)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11110461&assignment_repo_type=AssignmentRepo)

# Music Genre Classification
Our project aimed to conduct music analysis. We were provided with the archive fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip, which consisted of 8000 songs of 30s in mp3 format. Accompanied by a csv file containing relevant track information such as genre, album, listeners and artist, that can be found in TRACKS.csv.zip file. 

## Code structure
To start with, we have a main.ipynb file that it will import some files of the github repository as well as the TRACKS.csv.

As an experiment we tried to slice each spectrogram into time so that we were able to extract 20 partitions per each spectrogram because of data augmentation reasons. As we observed, a CNN model would give us an accuracy of 60%.

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
