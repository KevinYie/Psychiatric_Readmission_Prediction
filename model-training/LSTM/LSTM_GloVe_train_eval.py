#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import asarray
from numpy import zeros
from gensim import utils
import gensim.parsing.preprocessing as gsp

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

print('***Preprocessing data using tokenizer...')'

#Creating training and test
X_train, y_train = dtrain['TEXT'], dtrain['READMIT']
X_val, y_val  = dval['TEXT'], dval['READMIT']
X_test, y_test = dtest['TEXT'], dtest['READMIT']

#Tokenize inputs
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train =tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)

#Padding for same input length
vocab_size = len(tokenizer.word_index) + 1
maxlen = 350

X_train = pad_sequences(X_train, padding = "post", maxlen = maxlen)
X_test = pad_sequences(X_test, padding = "post", maxlen = maxlen)
X_val = pad_sequences(X_val, padding = "post", maxlen = maxlen)
# Reading in GloVe embeddings
embeddings_dictionary = dict()
glove_file = open(main_path+'glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

print('***Creating embedding matrix with our words...')
embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128, dropout_W = 0.3, dropout_U = 0.3))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0,
                    validation_data=(X_val, y_val))

torch.save(model,main_path+'model_LSTM_GloVe.pth')

print('***Evaluating on test set...')
y_proba = model.predict(X_test)
pickle.dump(y_proba,open(main_path+"model_lstm_glove_yproba.pkl","wb"))
