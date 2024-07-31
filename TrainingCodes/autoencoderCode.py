import argparse
import glob
import os
import random
import shutil
import sys

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from lightningModelAutoencoder import *
from layersAutoencoder import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Autoencoder Training Script")
    parser.add_argument('--gpu', type=int, required=True, help='GPU selection')
    parser.add_argument('--pval', type=int, choices=[1, 3], required=True, help='P-value selection (1 for p005, 3 for p0005)')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--chr', type=int, required=True, help='Chromosome number')
    parser.add_argument('--batch', type=int, required=True, help='Batch size')
    parser.add_argument('--epoch', type=int, required=True, help='Number of epochs')
    parser.add_argument('--decay', type=float, required=True, help='Weight Decay')
    parser.add_argument('--disease', type=str, required=True, help='Disease selection (Hypertension, T2D The name of the folder of the disease )')
    parser.add_argument('--CV', type=int, required=True, help='Cross-validation fold')
    parser.add_argument('--layer', type=int, required=True, help='Layer selection')
    parser.add_argument('--root_path', type=str, required=True, help='Root path for saving models')
    parser.add_argument('--log_path', type=str, required=True, help='Log path for TensorBoard logs')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    return parser.parse_args()

# The data should be saved as seperate numpy files under datapath folder.
class GenDataset(Dataset):
    def __init__(self, dataset_path, train, CVval, chr_num, p_val, disease):
        disease_name = disease
        mode = "Train" if train == 1 else "Test"
        self.file_list = glob.glob(f"{dataset_path}/{disease_name}/CV{CVval}/SeperateChrFiles/Chr{chr_num}/Scale/{mode}/*")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        x = torch.tensor(np.load(self.file_list[index]))
        return x


def main():
    args = parse_arguments()

    p_val = "p005" if args.pval == 1 else "p0005"
    disease_name = args.disease

    print(f"Started for {disease_name}, CV: {args.CV}, Pval: {p_val}, Chr: {args.chr}")

    random.seed(42)
    torch.manual_seed(42)

    train_dataset = GenDataset(args.dataset_path, 1, args.CV, args.chr, p_val, args.disease)
    test_dataset = GenDataset(args.dataset_path, 0, args.CV, args.chr, p_val, args.disease)
    
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Test Dataset Length: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=32)
    val_loader = DataLoader(test_dataset, batch_size=args.batch * 2, shuffle=False, num_workers=32)

    dummy = next(iter(val_loader))
    layers = createLayers(dummy, args.layer)[0]
    model_name = f"Model-{disease_name}-CV-{args.CV}-{args.layer}-{p_val}-Moduler-Autoencoder-{args.chr}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.root_path,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=model_name + '-{epoch}-{val_loss:.3f}--{train_loss:.3f}'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.log_path,
        name="Experiment-FCNN"
    )

    model = lightningModule(
        lr=args.lr,
        batch_size=args.batch,
        in_features=dummy.shape[1],
        layers=layers,
        p_val=p_val,
        max_epochs=args.epoch,
        disease=args.disease,
        CV=args.CV,
        w_decay=args.decay
    )

    trainer = L.Trainer(
        check_val_every_n_epoch=20,
        max_epochs=args.epoch,
        accelerator="gpu",
        strategy=DDPStrategy(),
        devices=[args.gpu],
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()



# Sample Usage

# python /home/ozdemiro/Codes/GitHub/autoencoderCode.py \
#     --gpu 0 \
#     --pval 1 \
#     --lr 0.001 \
#     --chr 1 \
#     --batch 32 \
#     --epoch 50 \
#     --decay 0.00001 \
#     --disease "T2D" \
#     --CV 1 \
#     --layer 1 \
#     --root_path "./Models/Autoencoder/" \
#     --log_path "./Logs/AutoencoderExperiment/" \
#     --dataset_path "/home/ozdemiro/LAB/learningData/chrData/FirstExperiments/"
