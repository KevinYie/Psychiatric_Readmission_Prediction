#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

import transformers
from transformers import BertTokenizer, AutoTokenizer, 
from transformers import BertModel, AutoModel
from transformers.optimization import AdamW

import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import copy

import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

'''
The hyperparameters used in this code are the best config found through hyper parameter tuning.
The saved model can be accessed through link in .txt file.
'''

#Check if CUDA is available
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
    
print('*** Loading data...')
main_path = '/scratch/mtp363/myjupyter/Project/'
dtrain_path = main_path+'project_dtrain.csv'
dval_path = main_path+'project_dval.csv'
dtest_path = main_path+'project_dtest.csv'

dtrain = pd.read_csv(dtrain_path, index_col = 0)
dval = pd.read_csv(dval_path, index_col = 0)
dtest = pd.read_csv(dtest_path, index_col = 0)

#Preprocess data using BERT tokenizer
print('*** Running bert_load for BERT Discharge...')
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
def bert_load(data):
    '''
    Load in data
    Return BERT's preprocessed inputs including token_id, mask, label
    '''
    token_ids = []
    attention_masks = []
    for row in data['TEXT']:
        row = re.sub('[\\(\[#.!?,\'\/\])0-9]', ' ', row)
        encoded_dict = tokenizer.encode_plus(row,
                                            add_special_tokens= True, #add [CLS], [SEP]
                                            max_length = 512,  
                                            pad_to_max_length = True, #pad and truncate
                                            return_attention_mask = True, #construct attention mask
                                            return_tensors = 'pt') #return pytorch tensor
        
        token_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    token_ids = torch.cat(token_ids,dim=0)
    attention_masks = torch.cat(attention_masks,dim=0)
    labels = torch.tensor(data['READMIT'].values)
    data_out = TensorDataset(token_ids, attention_masks, labels)
    return data_out
        
datatrain = bert_load(dtrain)   
dataval = bert_load(dval)
datatest = bert_load(dtest)


BATCH_SIZE = 12
train_loaderB = DataLoader(datatrain,
                           batch_size=BATCH_SIZE,
                           shuffle=True)
                           

val_loaderB = DataLoader(dataval,
                         batch_size=BATCH_SIZE,
                         shuffle= True)
                         

test_loaderB = DataLoader(datatest,
                         batch_size=BATCH_SIZE,
                         shuffle= False)

torch.manual_seed(2020)

print('*** Definining class BertClassification...')
class BertClassification(nn.Module):
  
    def __init__(self):
        super(BertClassification, self).__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        for param in self.bert.parameters():
            param.requires_grad = False
        nn.init.xavier_normal_(self.classifier.weight)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids,  attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs

modelBERT = BertClassification()
modelBERT.to(device)

def trainBERT(model, train_loader, val_loader, num_epoch=5, lr=2e-2 ):
    # Training steps
    start_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, eps= 1e-8) 
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    auc = []
    best_auc = 0.
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(num_epoch):
        model.train()
        #Initialize
        correct = 0
        total = 0
        total_loss = 0
     
        for i, (data, mask, labels) in enumerate(train_loader):
            data, mask, labels = data.to(device), mask.to(device), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()

            outputs = model(data, token_type_ids = None,
                                  attention_mask= mask,
                                  labels =None)
            
            loss = loss_fn(outputs.view(-1,2), labels.view(-1))
         

            loss.backward()
            optimizer.step()
            label_cpu = labels.squeeze().to('cpu').numpy()
            pred = outputs.data.max(-1)[1].to('cpu').numpy()
            total += labels.size(0)
            correct += float(sum((pred ==label_cpu)))
            total_loss += loss.item()
            
            
        acc = correct/total
       
        t_loss = total_loss/total
        train_loss.append(t_loss)
        train_acc.append(acc)
        # report performance
        
        print('Epoch: ',epoch)
        print('Train set | Accuracy: {:6.4f} | Loss: {:6.4f}'.format(acc, t_loss))     
    
    # Evaluate after every epoch
        #Reset the initialization
        correct = 0
        total = 0
        total_loss = 0
        model.eval()
        
        predictions =[]
        truths= []

        with torch.no_grad():
            for i, (data, mask, labels) in enumerate(val_loader):
                data, mask, labels = data.to(device), mask.to(device), labels.to(device, dtype=torch.long)


                optimizer.zero_grad()

                outputs = model(data, token_type_ids = None,
                                      attention_mask= mask,
                                      labels =None)
                #va_loss = loss_fn(outputs.squeeze(-1), labels)
                va_loss = loss_fn(outputs.view(-1,2), labels.view(-1))

                label_cpu = labels.squeeze().to('cpu').numpy()
                
                pred = outputs.data.max(-1)[1].to('cpu').numpy()
                total += labels.size(0)
                correct += float(sum((pred ==label_cpu)))
                total_loss += va_loss.item()
                
                predictions += list(pred)
                truths += list(label_cpu)
                       
            v_acc = correct/total
            v_loss = total_loss/total
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            
            
            v_auc = roc_auc_score(truths, predictions)
            auc.append(v_auc)
            
            elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
            print('Validation set | Accuracy: {:6.4f} | AUC: {:6.4f} | Loss: {:4.2f} | time elapse: {:>9}'.format(
                v_acc, v_auc, v_loss, elapse))
            print('-'*10)
            
            if v_auc > best_auc:
                best_auc = v_auc
                best_model = copy.deepcopy(model.state_dict())

    print('Best validation auc: {:6.4f}'.format(best_auc))
    model.load_state_dict(best_model)     
    return train_loss, train_acc, val_loss, val_acc, v_auc, model

print('*** Training BERT Discharge feature extractor...')
train_loss_BERTC, train_acc_BERTC, val_loss_BERTC, val_acc_BERTC, val_auc_BERTC, model_BERTC = trainBERT(modelBERT, 
                                                                                     train_loaderB, 
                                                                                     val_loaderB) 
                                                                                     
print('*** Saving best model...')
torch.save(model_BERTC,'bert_clinic_fc_2e2.pth')
