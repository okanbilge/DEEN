
import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import pytorch_lightning as L
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from pytorch_lightning.callbacks import ModelCheckpoint


class modulerNetwork2Layer(nn.Module):
    def __init__(self,sizeT):
    
        super(modulerNetwork2Layer, self).__init__()        

        self.fc0 = nn.Linear(sizeT[0], sizeT[1])
        self.fc1 = nn.Linear(sizeT[1], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu((self.fc0(x)))
        x = self.fc1(x)
        return x
         
class modulerNetwork3Layer(nn.Module):
    def __init__(self,sizeT):
    
        super(modulerNetwork3Layer, self).__init__()        

        self.fc0 = nn.Linear(sizeT[0], sizeT[1])
        self.fc1 = nn.Linear(sizeT[1], sizeT[2])
        self.fc2 = nn.Linear(sizeT[2], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu((self.fc0(x)))
        x = self.relu((self.fc1(x)))
        x = self.fc2(x)
        return x
    
class modulerNetwork4Layer(nn.Module):
    def __init__(self,sizeT):
    
        super(modulerNetwork4Layer, self).__init__()        

        self.fc0 = nn.Linear(sizeT[0], sizeT[1])
        self.fc1 = nn.Linear(sizeT[1], sizeT[2])
        self.fc2 = nn.Linear(sizeT[2], sizeT[3])
        self.fc3 = nn.Linear(sizeT[3], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu((self.fc0(x)))
        x = self.relu((self.fc1(x)))
        x = self.relu((self.fc2(x)))
        x = self.fc3(x)
   
        return x




class modulerNetwork5Layer(nn.Module):
    def __init__(self,sizeT):
    
        super(modulerNetwork5Layer, self).__init__()        
        self.fc0 = nn.Linear(sizeT[0], sizeT[1])
        self.fc1 = nn.Linear(sizeT[1], sizeT[2])
        self.fc2 = nn.Linear(sizeT[2], sizeT[3])
        self.fc3 = nn.Linear(sizeT[3], sizeT[4])
        self.fc4 = nn.Linear(sizeT[4], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x
         
                                
    



class lightningModule(L.LightningModule):
    def __init__(self,lr,batchSize,w_decay,inputSize,databasePath,layerInfo,pval,epoch,dataType,disease,CV=1):
        super().__init__()
        
        self.save_hyperparameters()

        self.inputSize = inputSize
        if len(layerInfo)==2:
            self.model = modulerNetwork2Layer(layerInfo)
        elif len(layerInfo)==3:
            self.model = modulerNetwork3Layer(layerInfo)
        elif len(layerInfo)==4:
            self.model = modulerNetwork4Layer(layerInfo)
        elif len(layerInfo)==5:
            self.model = modulerNetwork5Layer(layerInfo)
            
        self.weight_decay = w_decay
        self.lr = lr
        self.val_step_outputs = []        
        self.val_step_targets = [] 
        self.training_step_outputs = []   
        self.training_step_targets = []  


    def forward(self, x):
        
        
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y_hat =  self.forward(x.float())

        loss = nn.BCEWithLogitsLoss()(y_hat, y.float().view(-1, 1))


        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        
        y_pred = y_hat.detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        
        
        self.training_step_outputs.extend(y_pred)
        self.training_step_targets.extend(y_true)
       
        return loss
    def on_train_epoch_end(self):
        # Get AuC and define the errors if any.
        train_all_outputs = self.training_step_outputs
        train_all_targets = self.training_step_targets

        predictionsAll = np.asarray(train_all_outputs)
        decisions = np.max(predictionsAll,axis=1)
        labels = np.asarray(train_all_targets)
        auc = 0
        avg_auc = roc_auc_score(labels, predictionsAll)
        self.log('train_auc', avg_auc, prog_bar=True, logger=True, sync_dist=True)



        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        # free up the memory
      
        self.training_step_outputs.clear()
        self.training_step_targets.clear()


    def validation_step(self, batch, batch_idx):
        x, y  = batch
        y_hat =  self.forward(x.float())

        loss = nn.BCEWithLogitsLoss()(y_hat, y.float().view(-1, 1))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    


        y_pred = y_hat.detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()

        
        self.val_step_outputs.extend(y_pred)
        self.val_step_targets.extend(y_true)

  
        return {"val_loss": loss}

    def on_validation_epoch_end(self):

        val_all_outputs = self.val_step_outputs
        val_all_targets = self.val_step_targets

        predictionsAll = np.asarray(val_all_outputs)
        decisions = np.max(predictionsAll,axis=1)
        
        labels = np.asarray(val_all_targets)
        try:
            if self.current_epoch!=0:
                avg_auc = roc_auc_score(labels, predictionsAll)
                self.log('val_auc', avg_auc, prog_bar=True, logger=True, sync_dist=True)
            else:                    
                self.log('val_auc', 0, prog_bar=True, logger=True, sync_dist=True)

        except:
            print("AUC Error!") 

        self.val_step_outputs.clear()
        self.val_step_targets.clear()
   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

