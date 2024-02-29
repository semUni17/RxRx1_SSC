# RxRx1_SSC

This repo allows to train and test a self-supervised method for classification
of microscopy images taken from RxRx1 dataset.

Fundamental steps:
- train features extractor on RxRx1 dataset, with SimCLR framework
- test features extractor on RxRx1, visualizing the embedding
- train classifier on RxRx1 dataset, on top of features extractor
- test classifier on RxRx1 dataset

SimCLR code is adapted from https://github.com/sthalles/SimCLR


## :grey_question: Index
- [RxRx1_SSC](#RxRx1_SSC)
    - [Index](#index)
    - [Installation](#installation)
    - [Dependencies](#dependencies)
    - [Usage](#usage)


## :receipt: Installation
```
git clone https://github.com/semUni17/RxRx1_SSC.git
cd RxRx1_SSC
```


## :package: Dependencies
```
conda env create -f environment.yml
```


## :zap: Usage
To run the train and test file:
- `python src/processes/launch.py -n net -m mode -cfg path/to/config.yml`

Args for train and test file:
- `-n` (`--net`): choose network to train or test (simclr or classifier)
- `-m` (`--mode`): choose mode (train or test)
- `-cfg` (`--config_file`): config `.yml` file path (templates are already provided)
