# RiboDetector
Fast and accurate rRNA sequence detection using bi-directional LSTM

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [RiboDetector](#ribodetector)
  - [Requirements](#requirements)
  - [Usage](#usage)
    - [Config files](#config-files)
      - [`config.json` for training](#configjson-for-training)
      - [`predict_config.json` for training](#predictconfigjson-for-training)
    - [Training](#training)
    - [Prediction](#prediction)
  - [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.7 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm 


## Usage
Before you run, please modify the `config.json` for training and `predict_config.json` for prediction to adapt to your input and environment.


### Config files

Config files are in `.json` format:
#### `config.json` for training
```json
{
    "name": "Sequence_clf", # Name of the classifier
    "n_gpu": 1, # number of GPU to use

    "arch": {
        "type": "SeqModel", # Type os the model, we have only SeqModel here
        "args": {
            "input_size": 4, # The width of the input
            "hidden_size": 128, # number of hidden units for LSTM
            "num_layers": 1, # number of layers
            "num_classes": 2, # number of classes, here we have 2
            "batch_first": true, # batch is the first dimension
            "bidirectional": true # is bi-direction LSTM?
        }
    },
    "data_loader": {
        "type": "SeqDataLoader",
        "args": {
            "seq_data": { # here to define the path to input sequences
                "0": "../datasets/non-rrna-train-1m.fasta",
                "1": "../datasets/rrna-train-1m.fasta"
            },
            "min_seq_len": 100, # the sequence length of the input
            "batch_size": 1024,
            "shuffle": true, 
            "validation_split": 0.1, # the fraction of the input as validation set
            "num_workers": 2 # number of threads for data loading
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.001, # learning rate
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": "cross_entropy",
    "metrics": [ # metrics to report during training
        "recall", "accuracy", "precision", "F1", "mcc"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 20, # number of epochs

        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,

        "monitor": "min val_loss",
        "early_stop": 10,

        "tensorboard": false
    }
}
```
#### `predict_config.json` for training
```json
{
    "name": "Sequence_clf",
    "n_gpu": 1,
    "arch": {
        "type": "SeqModel",
        "args": { # please keep this the same with the trained model
            "input_size": 4,
            "hidden_size": 128,
            "num_layers": 1,
            "num_classes": 2,
            "batch_first": true,
            "bidirectional": true
        }
    },
    "state_file": {
        "read_len50": "saved_model/trained_1m_rl50.pth",
        "read_len100": "saved_model/trained_1m_rl100.pth",
        "read_len150": "saved_model/trained_1m_rl150.pth"
    }
}
```

### Training
Modify the configurations in `config.json`, then run:

  ```
  python train.py --config config.json
  ```

More options:
```
usage: train.py [-h] [-c CONFIG] [-r RESUME] [-d DEVICE] [--lr LR] [--bs BS]

DNA sequence classifier

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
  --lr LR, --learning_rate LR
  --bs BS, --batch_size BS
```

### Prediction
Modify the configurations in `predict_config.json` before run `predict.py`

  ```
  usage: predict.py [-h] [-c CONFIG] [-d DEVICEID] -l LEN -i [INPUT [INPUT ...]]
                  -o [OUTPUT [OUTPUT ...]] [-r [RRNA [RRNA ...]]]
                  [-e {rrna,norrna,both}] [--cpu]

rRNA sequence detector

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path
  -d DEVICEID, --deviceid DEVICEID
                        indices of GPUs to enable. Quotated comma-separated
                        device ID numbers. (default: all)
  -l LEN, --len LEN     Sequencing read length, should be not smaller than 50.
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        path to input sequence files, the second file will be
                        considered as second end if two files given.
  -o [OUTPUT [OUTPUT ...]], --output [OUTPUT [OUTPUT ...]]
                        path to the output sequence files after rRNAs removal
                        (same number of files as input)
  -r [RRNA [RRNA ...]], --rrna [RRNA [RRNA ...]]
                        path to the output sequence file of detected rRNAs
                        (same number of files as input)
  -e {rrna,norrna,both}, --ensure {rrna,norrna,both}
                        Applicable when given paired end reads; norrna: remove
                        as many as possible rRNAs, output non-rRNAs with high
                        confidence; rrna: vice versa, rRNAs with high
                        confidence; both: both non-rRNA and rRNA prediction
                        with high confidence
  --cpu                 Use CPU even GPU is available. Useful when GPUs and
                        GPU RAM are occupied.
  ```



## Acknowledgements
This project is based on the template [pytorch-template
](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque) and other [contributors](https://github.com/victoresque/pytorch-template/graphs/contributors).
