import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMTagger import LSTMTagger
import preprocess
from MyDataset import MyDataset
import time

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 1500
    NUM_EPOCHS = 1
    LAYER_NUM = 1
    BATCH_SIZE= 128
    tag_to_idx, idx_to_tag = preprocess.get_tag_dicts()
    #val_to_idx, idx_to_val = preprocess.get_dict_val()

    data = torch.Tensor([(tag_to_idx[(i[0],i[2],i[3])], 0) for i in (preprocess.get_small_dataset()[:250000])])
    data_val = torch.Tensor([(tag_to_idx[(i[0],i[2],i[3])], 0) for i in (preprocess.get_small_dataset()[250000:])])
    training_data = torch.utils.data.DataLoader(MyDataset(data, 50), BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)
    test_data = None
    val_data = torch.utils.data.DataLoader(MyDataset(data_val, 50), BATCH_SIZE, shuffle=False, drop_last=True, num_workers=8)

    model_tag = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tag_to_idx), len(tag_to_idx), LAYER_NUM)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model_tag.parameters())


    #-----------putting everything on GPU---------
    model_tag.cuda()
    #---------------------------------------------



    for epoch in range(NUM_EPOCHS): 

        model_tag.train()
        start_time = time.time()
        loss_sum = 0
        cnt = 0
        for sentence in training_data:
            
            size = int(sentence.size(0))
            model_tag.zero_grad()
            sentence_tag = sentence[:, :-1, 0].to(device)
            y_tag = sentence[:, -1, 0].to(device)

            y_pred_tag = model_tag(sentence_tag)
            
            correct = (y_pred_tag.argmax(dim=1) == y_tag).sum().item()

            loss = loss_function(y_pred_tag, y_tag.long())
            

            loss_sum+=loss
            cnt+=1
            if(cnt%10 == 0):
                print(f"current number of batches{cnt}, loss: {loss}", f'accuracy:{100*(correct/size)}')
                
            loss.backward()
            nn.utils.clip_grad_value_(model_tag.parameters(), 5.0)
            optimizer.step()

        
        model_tag.eval()
        correct = 0
        loss_sum = 0
        cnt = 0
        ep_cnt = 0
        with torch.no_grad():
            for sentence in val_data:

                sentence_tag = sentence[:, :-1, 0].to(device)
                y_tag = sentence[:, -1, 0].to(device)

                y_pred_tag = model_tag(sentence_tag)
                
                correct += (y_pred_tag.argmax(dim=1) == y_tag).sum().item()

                loss = loss_function(y_pred_tag, y_tag.long())
                loss_sum+=loss

                ep_cnt+=1
                cnt+= y_tag.size(0)
            print(f"Validation : loss {loss_sum/ep_cnt}, accuracy:{100* correct/cnt}")
        print(f'Epoch ended, time taken {time.time()-start_time}s')
                

    torch.save(model_tag, 'D://data//first_model_tag1.pickle')

if __name__ == "__main__":
    train()