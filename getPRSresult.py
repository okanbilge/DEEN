import numpy as np
import torch
from ModelCodes.autoencoderModelv2 import *
from ModelCodes.classificationModelv1 import *
import pickle
import os
import tqdm
from sklearn.metrics import roc_curve, auc
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Autoencoder Training Script")
    parser.add_argument('--disease', type=str, required=True, help='Disease name')
    parser.add_argument('--pheno', type=str, required=False, help='Phenotype file if exists')
    parser.add_argument('--save', type=str, required=False, help='Numpy path if want to save')

    return parser.parse_args()


def getAEdata(diseaseName,chrIndex):

    file_name = "/home/ozdemiro/Codes/GitHub/Data/RawFiles/"+diseaseName+"/Chr"+str(chrIndex)+".raw"
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Data file {file_name} not found")

    with open(file_name, 'r') as file:
        data = file.read().replace("NA", "0").strip().split("\n")[1:]


    processed_data = np.array([np.array(row.split("\t")[6:], dtype=np.uint8) for row in data])


    scaler_name = "/home/ozdemiro/Codes/GitHub/Data/Scalers/"+diseaseName+"/Chr"+str(chrIndex)+"-Scaler.pkl"
    with open(scaler_name, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    processed_data = scaler.transform(processed_data)

    modelPath = "/home/ozdemiro/Codes/GitHub/Models/"+diseaseName+"/Autoencoder/Autoencoder-"+str(chrIndex)+".ckpt"
    model = lightningModule.load_from_checkpoint(modelPath,map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        output = model.encoder(torch.from_numpy(processed_data).float())
    return output


def main():
    args = parse_arguments()
    diseaseName = args.disease
    print("Autoencoder Started...")
    for chrIndex in tqdm.tqdm(range(1,23)):

        if chrIndex ==1:
            autoencoder_file = getAEdata(diseaseName,chrIndex)
        else:
            autoencoder_file = torch.cat((autoencoder_file,getAEdata(diseaseName,chrIndex)),dim=1)

    print("Autoencoder Finished...")
    print("Classification Started...")

    classificationModel = "/home/ozdemiro/Codes/GitHub/Models/"+diseaseName+"/Classification/"+diseaseName+"-Classification.ckpt"
    model = classificationModule.load_from_checkpoint(classificationModel,map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        output = model(autoencoder_file)
    print("Classification Finished...")


    
    if args.pheno!=None:
        if os.path.exists(args.pheno):
            print("Checking performance numbers")
            with open(args.pheno) as file:
                phenoFile = file.read()
                phenoFile = phenoFile.split("\n")
                phenoFile=phenoFile[1:]
            labels = []
            for x in range(len(phenoFile)):
                labels.append(int(phenoFile[x].split(" ")[2])-1)
            labels = np.asarray(labels)

            fpr, tpr, thresholds = roc_curve(labels, output[:,1])
            roc_auc = auc(fpr, tpr)

            print(f"AUC: {roc_auc:.2f}")
        else:
            print("Phenotype not exists...")

    if args.save != None:
        np.save(args.save,output)


if __name__ == "__main__":
    main()


# Sample code to run
# python /home/ozdemiro/Codes/GitHub/getPRSresult.py --disease Hypertension --pheno ./Data/PhenoFiles/Hypertension --save .outputResult.npy