
# RAST
This code is a PyTorch implementation of our paper "RAST".

## Requirements
We implement the experiments on a Linux Server with CUDA 12.2 equipped with 4x A6000 GPUs. For convenience, execute the following command.
```
# Install Python
conda create -n RAST python==3.11
conda activate RAST

# Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install other dependencies
pip install -r requirements.txt
```

## Dataset
PEMS04
You can download the dataset from [LargeST](https://github.com/liuxu77/LargeST) or [BasicTS](https://github.com/GestaltCogTeam/BasicTS/blob/master/tutorial/getting_started.md). Unzip the files to the datasets/ directory:

## Quick Start
```bash
## Pretrain Baselines
python experiments/train.py -c baselines/D2STGNN/SD.py -g 2

# train RAST from scratch
nohup python experiments/train.py -c src/train_SD.py -g 2 > logs/test.txt &

# Retrieval-Augmented pre-trained STGNNs
bash run_rast.sh

## Analysis
cd src
python analyze_retrieval_store.py --file=../database/SD_store_epoch_1.npz

```
