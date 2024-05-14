# LyS at SemEval 2024: Multimodal Cause Emotion Extraction in Multi-Party Conversations 💁🏻‍♂️💭🙋🏻‍♀️

Hi!👋 This repository contains the code developed for our proposal *"[An Early Prototype for End-to-End Emotion
Linking as Graph-Based Parsing](https://arxiv.org/abs/2405.06483)"* for the [SemEval 2024 (Task 3)](https://nustm.github.io/SemEval-2024_ECAC/). 

## Requirements ⚙️

Our models are entirely implemented in [Python 3.10.7](https://www.python.org/downloads/release/python-3107/) with the following libraries:

- [PyTorch 2.2.0](https://pytorch.org/).
- [Transformers 4.37.2](https://huggingface.co/docs/transformers/index).
- [Torchvision 0.17.0](https://pytorch.org/vision/stable/index.html).
- [Torchaudio 2.2.0](https://pytorch.org/audio/stable/index.html).

We strongly recommend fine-tuning the models with a GPU device, so firstly you might need to check [PyTorch + CUDA compatibility](https://pytorch.org/get-started/previous-versions/). To train the multimodal encoder our code supports multi-device mapping, so it is possible to configurate the distributionn of the pretrained models across different GPU devices. 

To automatically install all the packages, run the [requirements.txt](requirements.txt) file via: 

```shell
pip3 install -r requirements.txt
```

## Data preparation 🛠️

The train and validation sets used to conduct our experiments can be found in the folder [dataset/](dataset/). The folder [dataset/text/](dataset/text/) stores the main JSON files to conduct both subtasks and, at first instance, the folders [dataset/video](dataset/video/)  and [dataset/audio/](dataset/audio/) will be empty. To properly prepare the data, ensure the following data folder structure:

```
dataset/    
    text/
        Subtask_1_trainset.json     # train split from Subtask_1_train.json
        Subtask_1_devset.json       # dev split from Subtask_1_train.json
        Subtask_1_test.json         # official test split 
        Subtask_1_train.json        # official train split 
        Subtask_2_trainset.json     # train split from Subtask_2_train.json
        Subtask_2_devset.json       # dev split from Subtask_2_train.json
        Subtask_2_test.json         # official test split 
        Subtask_2_train.json        # official train split 
    video/
        dia*ut*.mp4                 # all videos
        ...
    audio/
        dia*ut*.wav                 # all audios
```

The only data that is not directly downloaded from the [official statement of the task](https://nustm.github.io/SemEval-2024_ECAC/) is the [dataset/audio/](dataset/audio) folder. Previously storing the audios will permit to fastly train the models and afford some extra repeated computations (loading audios from .mp4 has a higher computational cost than loading from WAV). To automatically get the [dataset/audio/](dataset/audio/) folder, once all the videos are loaded in [dataset/video/](dataset/video/), it is possible to run the [audio.py](audio.py) script. This script will automatically produce all .wav files.

```
python3 audio.py
```


## Subtask 1. Textual Cause Emotion Extraction ✍️

The script [subtask1.py](subtask1.py) executes the system described for the first subtask. The model configuration can be fixed with an INI file (see [config/subtask1.ini](config/subtask1.ini) for default configuration). By default, results will be stored and loaded from [results/subtask1/](results/subtask1/) folder and the train, validation and test set used will be those specified above.

1. Training:

```shell 
python3 subtask1.py train 
```

2. Predict:

```shell 
python3 subtask1.py predict 
```

3. Evaluate:

```shell 
python3 subtask1.py eval
```


## Subtask 2. Multimodal Cause Emotion Extraction 🗣️
The script [subtask2.py](subtask2.py) executes the system described for the second subtask. The model configuration can be fixed with an INI file (see [config/subtask2.ini](config/subtask2.ini) for default configuration). Suggested arguments to change (depending on the computational capabilities of your machine) are the pretrained vision and audio models, as well as the number of frames loaded per video and the embedding size of each modality.  By default, results will be stored and loaded from [results/subtask2/](results/subtask2/) folder and the train, validation and test set used will be those specified above.

1. Training:

```shell 
python3 subtask2.py train 
```

2. Predict:

```shell 
python3 subtask2.py predict 
```

3. Evaluate:

```shell 
python3 subtask2.py eval
```
