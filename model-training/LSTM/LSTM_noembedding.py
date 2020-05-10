#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import time


'''
The hyperparameters used in this code are the best config found through hyper parameter tuning.
The saved model can be accessed through link in .txt file.
'''
#Check CUDA resource
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

print('***Preprocessing data using tokenizer...')
UNK = "<UNK>"
PAD = "<PAD>"

def build_vocab(data, min_count=3, max_vocab=None):
    """
    Build vocabulary from sentences (list of strings)
    """
    # keep track of the number of appearance of each word
    word_count = Counter()
    data = data.astype(str)
    for i in range(len(data)):
        sentence = re.sub('[\\(\[:;*#.!?,\'\/\])0-9]', ' ',data.iloc[i])
        word_count.update(word_tokenize(sentence.lower()))
    
    vocabulary = list([w for w in word_count if word_count[w] > min_count]) + [UNK, PAD]
    indices = dict(zip(vocabulary, range(len(vocabulary))))

    return vocabulary, indices

vocabulary, vocab_indices = build_vocab(dtrain['TEXT'])

class LoadDataset(Dataset):
    def __init__(self, vocab_index, data, label = 'READMIT'):
        self.vocab_index = vocab_index
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        self.data['TEXT'].astype(str)
        sentence = self.data['TEXT'].iloc[idx]
      
        sentence = re.sub('[\\(\[#.!?,\'\/\])0-9]', ' ', sentence)

        token_indices = np.array([self.vocab_index[word] if word in self.vocab_index else self.vocab_index['<UNK>'] 
                                  for word in word_tokenize(sentence.lower())])
     
        return (torch.tensor(token_indices) , self.data['READMIT'].iloc[idx])


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=len(vocabulary)-1)

    return torch.as_tensor(xx_pad), torch.as_tensor(x_lens), torch.LongTensor(yy)



BATCH_SIZE = 32

train_loader = DataLoader(LoadDataset(vocab_indices, dtrain),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          collate_fn = pad_collate)

val_loader = DataLoader(LoadDataset(vocab_indices, dval),
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         collate_fn = pad_collate)

test_loader = DataLoader(LoadDataset(vocab_indices, dtest),
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         collate_fn = pad_collate)

print('*** Defining LSTM model')
class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, 
                 vocab_size, embedding_dim, rnn='LSTM'):
        super(LSTMModel, self).__init__()
        
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_size-1)
        self.hidden_dim = hidden_dim
        self.rnn_fn = rnn
        assert self.rnn_fn in ['GRU','LSTM']
        self.rnn = getattr(nn, rnn)(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, x_len):
        x = self.emb(x)
        
        _, last_hidden = self.rnn(pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False))
        if self.rnn_fn == 'LSTM':
            last_hidden = last_hidden[0]
        out = self.fc(last_hidden.view(-1, self.hidden_dim))
        out = self.softmax(out)
        return out
    
torch.manual_seed(2020)

def train(model, train_loader=train_loader, val_loader=val_loader, num_epoch=10):
    # Training steps
    start_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate=0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10**(-5))
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    best_acc = 0.
    best_model = model

    for epoch in range(num_epoch):
        model.train()
        #Initialize
        correct = 0
        total = 0
        total_loss = 0

        for i, (data, data_len, labels) in enumerate(train_loader):
            data, data_len, labels = data.to(device), data_len.to(device), labels.to(device)
            outputs = model(data, data_len)
            model.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = outputs.data.max(-1)[1]  
            total += labels.size(0)
            correct += torch.sum(pred ==labels).item()
            total_loss += loss.item()
        
        acc = correct/total
        t_loss = total_loss/total
        train_loss.append(t_loss)
        train_acc.append(acc)
        
        print('Epoch: ',epoch)
        print('Train set | Accuracy: {:6.4f} | Loss: {:6.4f}'.format(acc, t_loss))     
    
        # Evaluate after every epoch
        #Reset the initialization
        correct = 0
        total = 0
        total_loss = 0
        model.eval()
        
        predictions = []
        truths = []

        with torch.no_grad():
            for i, (data, data_len, labels) in enumerate(val_loader):
                data, data_len, labels = data.to(device), data_len.to(device), labels.to(device)
                outputs = model(data, data_len)
                va_loss = loss_fn(outputs, labels)
                pred = outputs.data.max(-1)[1]
            
                total += labels.size(0)
                correct += torch.sum(pred ==labels).item()
                total_loss += va_loss.item()
                
                predictions += list(pred.cpu().numpy())
                truths += list(labels.cpu().numpy())
                
            auc = roc_auc_score(truths, predictions)
                
                
            v_acc = correct/total
            v_loss = total_loss/total
            val_loss.append(v_loss)
            val_acc.append(v_acc)
        
            elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
            print('Validation set | Accuracy: {:6.4f} | AUC: {:4.2f} | Loss: {:4.2f} | time elapse: {:>9}'.format(
                v_acc, auc, v_loss, elapse))
            print('-'*10)
            
            if v_acc > best_acc:
                best_acc = v_acc
                best_model = model.state_dict()

    print('Best validation accuracy: {:6.4f}'.format(best_acc))
    model.load_state_dict(best_model)     
    return train_loss, train_acc, val_loss, val_acc, model
        
    
LSTM_model = LSTMModel(100,2,len(vocabulary),128).to(device)
train_loss_LSTM, train_acc_LSTM, val_loss_LSTM, val_acc_LSTM, model_LSTM = train(LSTM_model)

torch.save(model_LSTM,main_path+'model_LSTM_noGloVe.pth')

