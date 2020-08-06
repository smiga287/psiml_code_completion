import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMTagger import LSTMTagger
import preprocess
from MyDataset import MyDataset

EMBEDDING_DIM = 100
HIDDEN_DIM = 1500
NUM_EPOCHS = 2
LAYER_NUM = 1
BATCH_SIZE= 16
tag_to_idx, idx_to_tag = preprocess.get_tag_dicts()
#val_to_idx, idx_to_val = preprocess.get_dict_val()

data = [(tag_to_idx[(i[0],i[2],i[3])], 0) for i in preprocess.get_small_dataset()]

training_data = torch.utils.data.DataLoader(MyDataset(data, 50), BATCH_SIZE, shuffle=True)
test_data = None
val_data = None

model_tag = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tag_to_idx), len(tag_to_idx), LAYER_NUM)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model_tag.parameters())


#-----------putting everything on GPU---------
#model_tag.cuda()
#---------------------------------------------


for epoch in range(NUM_EPOCHS): 

    model_tag.train()

    loss_sum = 0
    cnt = 0
    for sentence in training_data:
        #todo: put imput to GPU and devide UNK from others

        model_tag.zero_grad()
        sentence_tag = sentence[:-1,0]
        y_tag = sentence[1:, 0]

        y_pred_tag = model_tag(sentence_tag)

        loss = loss_function(y_pred_tag, y_tag)

        loss_sum+=loss
        cnt+=1
        if(cnt%100 == 0):
            print(f"current number of batches{cnt}, loss: {loss/y_tag.size(0)}",'\n')
        loss.backward()
        nn.utils.clip_grad_value_(model_tag.parameters(), 5.0)
        optimizer.step()
    
    '''model_tag.eval()
    corect = 0
    loss_sum = 0
    cnt = 0
    with torch.no_grad():
        for sentence, y in val_data:
            y_pred = model_tag(sentence)
            loss = loss_function(y_pred, y)
            loss_sum += loss
            sol = y_pred.argmax(dim=1)
            corect += (sol == y).sum()
            cnt+= y.size(0)
        print(f"Validation : -loss {loss_sum/cnt}, accuracy:{100* corect/cnt}")'''
            
    
