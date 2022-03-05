# AudioClassification

This repository is used for audio classification task and provides training, prediction, prediction GUI, prediction Web GUI, hyperparameters tuning, and pretrained models.

## Installation

```bash
pip install -r requirements.txt
```

## Prepare dataset

Please prepare the dataset according to the following examples.

```
dataset
├── train   #for training
│   ├── class1
│   │   ├── wav1.wav
│   │   ├── wav2.wav
│   │   └── wav3.wav
│   ├── class2
│   │   ├── wav1.wav
│   │   ├── wav2.wav
│   │   └── wav3.wav
│   └── class3
│       ├── wav1.wav
│       ├── wav2.wav
│       └── wav3.wav
├── val     #for validation
│   ├── class1
│   │   ├── wav1.wav
│   │   ├── wav2.wav
│   │   └── wav3.wav
│   ├── class2
│   │   ├── wav1.wav
│   │   ├── wav2.wav
│   │   └── wav3.wav
│   └── class3
│       ├── wav1.wav
│       ├── wav2.wav
│       └── wav3.wav
└── test     #for testing
    ├── class1
    │   ├── wav1.wav
    │   ├── wav2.wav
    │   └── wav3.wav
    ├── class2
    │   ├── wav1.wav
    │   ├── wav2.wav
    │   └── wav3.wav
    └── class3
        ├── wav1.wav
        ├── wav2.wav
        └── wav3.wav
```

## Configuration

This repository provides the default configuration, which is [SPEECHCOMMANDS](config/config_SPEECHCOMMANDS.yml).

All parameters are in the YAML file.

## Argparse

You can override parameters by argparse while running.

```bash
python main.py --config config.yaml --str_kwargs mode=train #override mode as 100
python main.py --config config.yaml --num_kwargs max_epochs=100 #override training iteration as 100
python main.py --config config.yaml --bool_kwargs early_stopping=False #override early_stopping as False
python main.py --config config.yaml --str_list_kwargs classes=1,2,3 #override classes as 1,2,3
python main.py --config config.yaml --dont_check #don't check configuration
```

## Training

```bash
python main.py --config config.yml --str_kwargs mode=train # or you can set train as the value of mode in configuration
```

## Predict

```bash
python main.py --config config.yml --str_kwargs mode=predict,root=FILE # predict a file
python main.py --config config.yml --str_kwargs mode=predict,root=DIRECTORY # predict files in the folder
```

## Predict GUI

```bash
python main.py --config config.yml --str_kwargs mode=predict_gui    # will create a tkinter window
python main.py --config config.yml --str_kwargs mode=predict_gui --bool_kwargs web_interface=True   #will create a web interface by Gradio
```

## Tuning

```bash
python main.py --config config.yaml --str_kwargs mode=tuning    #the hyperparameter space is in the configuration
```

## Pretrained

This repository provides pretrained model. Please look at the pretrained directory.
