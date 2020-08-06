import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMTagger import LSTMTagger
import preparation

EMBEDDING_DIM = 100
HIDDEN_DIM = 1500
NUM_EPOCHS = 2
tag_to_idx, idx_to_tag = preparation.get_dict_tags()
val_to_idx, idx_to_val = preparation.get_dict_val()
training_data = None
test_data = None
val_data = None

model_tag = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tag_to_idx), len(tag_to_idx))
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model_tag.parameters())


#-----------putting everything on GPU---------
#model_tag.cuda()
#---------------------------------------------


for epoch in range(NUM_EPOCHS): 

    model_tag.train()

    loss_sum = 0
    cnt = 0
    for sentence, y in training_data:

        model_tag.zero_grad()

        y_pred = model_tag(sentence)

        loss = loss_function(y_pred, y)

        loss_sum+=loss
        cnt+=1
        if(cnt%100 == 0):
            print(f"current number of batches{cnt}, loss: {loss_sum/cnt}",'\n')
        loss.backward()
        optimizer.step()
    
    model_tag.eval()
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
        print(f"Validation : -loss {loss_sum/cnt}, accuracy:{100* corect/cnt}")
            
    