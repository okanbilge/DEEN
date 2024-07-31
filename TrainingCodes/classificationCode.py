import argparse
import glob
import os
import random
import sys
import csv
import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch import loggers as pl_loggers

from lightningModelClassification import *
from layers import *
from earlyStop import *


class MyEarlyStoppingCallback(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        validation_auc = pl_module.validation_auc_metric
        super().on_validation_end(trainer, pl_module)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Classification Training Script")
    parser.add_argument('--gpu', type=int, required=True, help='GPU selection')
    parser.add_argument('--pval', type=str, required=True, choices=['p005', 'p0005'], help='P-value selection')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--decay', type=float, required=True, help='Decay rate')
    parser.add_argument('--batch', type=int, required=True, help='Batch size')
    parser.add_argument('--epoch', type=int, required=True, help='Number of epochs')
    parser.add_argument('--dataType', type=int, required=True, choices=[0, 1, 2, 3], help='Data type selection')
    parser.add_argument('--disease', type=int, required=True, choices=[0, 1], help='Disease selection')
    parser.add_argument('--layer', type=int, required=True, help='Layer selection')
    parser.add_argument('--CV', type=int, required=True, help='Cross-validation fold')
    parser.add_argument('--rootPath', type=str, required=True, help='Root path for saving models')
    parser.add_argument('--dataPath', type=str, required=True, help='Path to dataset')
    return parser.parse_args()

class GenDataset(Dataset):
    def __init__(self, data_path, train=1, p_val="p0005", disease_name=1, CV=1, data_type_name='Normal'):
        file_list = []
        file_list.append(glob.glob(f"{data_path}/{disease_name}/CV{CV}/{p_val}/CombinedChrFiles/{data_type_name}/*_1_0_0_0_0_Data.npy"))
        file_list.append(glob.glob(f"{data_path}/{disease_name}/CV{CV}/{p_val}/CombinedChrFiles/{data_type_name}/*_0_1_0_0_0_Data.npy"))
        file_list.append(glob.glob(f"{data_path}/{disease_name}/CV{CV}/{p_val}/CombinedChrFiles/{data_type_name}/*_0_0_1_0_0_Data.npy"))
        file_list.append(glob.glob(f"{data_path}/{disease_name}/CV{CV}/{p_val}/CombinedChrFiles/{data_type_name}/*_0_0_0_1_0_Data.npy"))
        file_list.append(glob.glob(f"{data_path}/{disease_name}/CV{CV}/{p_val}/CombinedChrFiles/{data_type_name}/*_0_0_0_0_1_Data.npy"))

        if train == 1:
            self.file_list = []
            for x in range(1, 6):
                if x != CV:
                    self.file_list.extend(file_list[x - 1])
        else:
            self.file_list = file_list[CV - 1]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        x = torch.tensor(np.load(self.file_list[index]))
        file_info = self.file_list[index].split("_")
        y = torch.tensor(int(file_info[1][0]))
        return x, y

def main():
    args = parse_arguments()

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    if args.dataType == 0:
        dataTypeName = 'Normal'
    elif args.dataType == 1:
        dataTypeName = 'Scaled'
    elif args.dataType == 2:
        dataTypeName = 'PCA'
    elif args.dataType == 3:
        dataTypeName = 'Autoencoder'

    diseaseName = "Hypertension" if args.disease == 0 else "T2D"
    pVal = args.pval

    rootPath = os.path.join(args.rootPath, f"{diseaseName}-Classifier/CV{args.CV}/")  # Define Root Path Here
    dataPath = args.dataPath


    random.seed(42)
    torch.manual_seed(42)
    train_dataset = GenDataset(dataPath, 1, pVal, diseaseName, args.CV, dataTypeName)
    test_dataset = GenDataset(dataPath, 0, pVal, diseaseName, args.CV, dataTypeName)

    print("Train Loader Size", len(train_dataset))
    print("Test Loader Size", len(test_dataset))

    train, val = data.random_split(train_dataset, [0.90, 0.10])
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=32)
    val_loader = DataLoader(val, batch_size=args.batch * 2, shuffle=False, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=args.batch * 2, shuffle=False, num_workers=32)

    dummy = next(iter(train_loader))

    model_name = f"Model-FCNN-{args.CV}-{dataTypeName}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=rootPath,
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        filename=model_name + '--{epoch}--{val_loss:.3f}--{train_loss:.3f}--{train_auc:.3f}--{val_auc:.3f}'
    )

    layers[args.layer][0] = int(dummy[0].shape[1])
    print(layers[args.layer])

    early_stopping_callback = MultiCriterionEarlyStopping(patience=10)

    gpu_list = [args.gpu] if args.gpu < 8 else [0, 1, 2, 3]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=f"./Logs/tensorboardLogs-{diseaseName}-{dataTypeName}/",
        name=f"Experiment-FCNN-{args.CV}-{dataTypeName}-{pVal}"
    )

    model = lightningModule(
        lr=args.lr,
        batch_size=args.batch,
        decay=args.decay,
        in_features=dummy[0].shape[1],
        data_type_name=dataTypeName,
        layers=layers[args.layer],
        p_val=pVal,
        max_epochs=args.epoch,
        data_type=args.dataType,
        disease=args.disease,
        CV=args.CV
    )

    trainer = L.Trainer(
        max_epochs=args.epoch,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=gpu_list,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger
    )

    try:
        trainer.fit(model, train_loader, val_loader)
        best_checkpoint_path = checkpoint_callback.best_model_path
        best_model = lightningModule.load_from_checkpoint(best_checkpoint_path)
        best_model.eval()
        best_model.to("cuda:0")
        true_labels = []
        predicted_probs = []

        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                inputs, labels = batch
                outputs = best_model(inputs.float().to("cuda:0"))
                predicted_probs.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(true_labels, predicted_probs)
        print(f"Test AUC: {test_auc}")

        new_line = [diseaseName, args.CV, dataTypeName, args.CV, pVal, args.lr, args.decay, args.batch, best_checkpoint_path, test_auc, layers[args.layer]]
        print(new_line)
        with open('results.csv', 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(new_line)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()


# Sample Usage
# /home/ozdemiro/Codes/GitHub/classificationCode.py \
#     --gpu 0 \
#     --pval p005 \
#     --lr 0.001 \
#     --decay 0.001 \
#     --batch 32 \
#     --epoch 50 \
#     --dataType 0 \
#     --disease 0 \
#     --layer 1 \
#     --CV 1 \
#     --rootPath "/home/ozdemiro/Codes/Models/" \
#     --dataPath "/path/to/dataset/"
