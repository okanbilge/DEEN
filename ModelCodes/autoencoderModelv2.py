
import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import pytorch_lightning as L
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from pytorch_lightning.callbacks import ModelCheckpoint



    
class EncoderV1(nn.Module):
    def __init__(self,sizeT):
    
        super(EncoderV1, self).__init__()
        
        self.fc0 = nn.Linear(sizeT[0], sizeT[1])

        # self.relu = nn.LeakyReLU(negative_slope = 0.01)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc0(x))

        return x   
    

class EncoderV2(nn.Module):
    def __init__(self,sizeT):
    
        super(EncoderV2, self).__init__()
        
        self.fc0 = nn.Linear(sizeT[0], sizeT[1])
        self.fc1 = nn.Linear(sizeT[1], sizeT[2])
       
        # self.relu = nn.LeakyReLU(negative_slope = 0.01)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        return x

class EncoderV3(nn.Module):
    def __init__(self,sizeT):
    
        super(EncoderV3, self).__init__()
        
        self.fc0 = nn.Linear(sizeT[0], sizeT[1])
        self.fc1 = nn.Linear(sizeT[1], sizeT[2])
        self.fc2 = nn.Linear(sizeT[2], sizeT[3])
       
        # self.relu = nn.LeakyReLU(negative_slope = 0.01)
        self.relu = nn.ReLU()
    def forward(self, x):

        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x       
    
class EncoderV4(nn.Module):
    def __init__(self,sizeT):
    
        super(EncoderV4, self).__init__()
        
        self.fc0 = nn.Linear(sizeT[0], sizeT[1])
        self.fc1 = nn.Linear(sizeT[1], sizeT[2])
        self.fc2 = nn.Linear(sizeT[2], sizeT[3])
        self.fc3 = nn.Linear(sizeT[3], sizeT[4])
       
        # self.relu = nn.LeakyReLU(negative_slope = 0.01)
        self.relu = nn.ReLU()
    def forward(self, x):

        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x      
    
   

    
class DecoderV1(nn.Module):
    def __init__(self,sizeT):
    
        super(DecoderV1, self).__init__()
        


        self.fc0 = nn.Linear(sizeT[1], sizeT[0])
       
      
    def forward(self, x):

        x = self.fc0(x)
    
        return x
    
class DecoderV2(nn.Module):
    def __init__(self,sizeT):
    # def __init__(self,sizeT=192951):
    
        super(DecoderV2, self).__init__()
        

        self.fc0 = nn.Linear(sizeT[2], sizeT[1])
        self.fc1 = nn.Linear(sizeT[1], sizeT[0])    
        self.relu = nn.LeakyReLU(negative_slope = 0.01)

    def forward(self, x):

        x = self.relu(self.fc0(x))
        x = self.fc1(x)
    
        return x



class DecoderV3(nn.Module):
    def __init__(self,sizeT):
    
        super(DecoderV3, self).__init__()
        

        self.fc0 = nn.Linear(sizeT[3], sizeT[2])
        self.fc1 = nn.Linear(sizeT[2], sizeT[1])
        self.fc2 = nn.Linear(sizeT[1], sizeT[0])
       
        self.relu = nn.LeakyReLU(negative_slope = 0.01)

    def forward(self, x):

        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class DecoderV4(nn.Module):
    def __init__(self,sizeT):
    
        super(DecoderV4, self).__init__()
        

        self.fc0 = nn.Linear(sizeT[4], sizeT[3])
        self.fc1 = nn.Linear(sizeT[3], sizeT[2])
        self.fc2 = nn.Linear(sizeT[2], sizeT[1])
        self.fc3 = nn.Linear(sizeT[1], sizeT[0])
       
        self.relu = nn.LeakyReLU(negative_slope = 0.01)

    def forward(self, x):

        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    

class lightningModule(L.LightningModule):
    def __init__(self,lr,batchSize,w_decay,inputSize,databasePath,layerInfo,pval,epoch,dataType,disease):
        super().__init__()
        
        self.save_hyperparameters()

        self.inputSize = inputSize
        if len(layerInfo)==2:
            self.encoder = EncoderV1(layerInfo)
            self.decoder = DecoderV1(layerInfo)
        elif len(layerInfo)==3:
            self.encoder = EncoderV2(layerInfo)
            self.decoder = DecoderV2(layerInfo)
        elif len(layerInfo)==4:
            self.encoder = EncoderV3(layerInfo)
            self.decoder = DecoderV3(layerInfo)
        elif len(layerInfo)==5:
            self.encoder = EncoderV4(layerInfo)
            self.decoder = DecoderV4(layerInfo)

        self.weight_decay = w_decay
        self.lr = lr


    def forward(self, x):
        
        return self.decoder(self.encoder(x))     
    
      
    def training_step(self, batch, batch_idx):

        x  = batch
        y_hat =  self.forward(x.float())
        loss = nn.MSELoss()(y_hat, x.float())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    


    def validation_step(self, batch, batch_idx):


        x  = batch
        y_hat =  self.forward(x.float())
        loss = nn.MSELoss()(y_hat, x.float())        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)



        return {"val_loss": loss}

   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

