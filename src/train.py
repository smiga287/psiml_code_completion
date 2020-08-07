import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMTagger import LSTMTagger
from DataManager import DataManager, SMALL, TRAIN
from Dataset import Dataset
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 1500
    NUM_EPOCHS = 1
    LAYER_NUM = 1
    BATCH_SIZE = 512

    data_manager = DataManager(TRAIN)

    tag_to_idx, idx_to_tag = data_manager.get_tag_dicts()
    val_to_idx, idx_to_val = data_manager.get_val_dicts()

    data = torch.Tensor(
        [
            (tag_to_idx[(tag, have_children, have_sibling)], 0)
            for tag, _, have_children, have_sibling in (data_manager.get_data()[:5000000])
        ]
    )
    data_val = torch.Tensor(
        [
            (tag_to_idx[(tag, have_children, have_sibling)], 0)
            for tag, _, have_children, have_sibling in (data_manager.get_data()[5000000:55000000])
        ]
    )

    training_data = torch.utils.data.DataLoader(
        Dataset(data), BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8
    )
    test_data = None
    val_data = torch.utils.data.DataLoader(
        Dataset(data_val), BATCH_SIZE, shuffle=False, drop_last=True, num_workers=8
    )

    model_tag = LSTMTagger(
        EMBEDDING_DIM, HIDDEN_DIM, len(tag_to_idx), len(tag_to_idx), LAYER_NUM
    )
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model_tag.parameters())

    # -----------putting everything on GPU---------
    model_tag.cuda()
    # ---------------------------------------------

    for epoch in range(NUM_EPOCHS):

        summary_writer = SummaryWriter()

        model_tag.train()
        start_time = time.time()
        loss_sum = 0
        cnt = 0
        for i, sentence in tqdm(
            enumerate(training_data),
            total=len(training_data),
            desc=f"Epoch: {epoch}",
            unit="batches",
        ):
            global_step = epoch * len(training_data) + i
            size = int(sentence.size(0))
            model_tag.zero_grad()
            sentence_tag = sentence[:, :-1, 0].to(device)
            y_tag = sentence[:, -1, 0].to(device)

            y_pred_tag = model_tag(sentence_tag)

            correct = (y_pred_tag.argmax(dim=1) == y_tag).sum().item()

            loss = loss_function(y_pred_tag, y_tag.long())
            summary_writer.add_scalar("Train loss", loss, global_step)

            loss_sum+=loss
            cnt+=1
            #if(cnt%10 == 0):
                #print(f"current number of batches{cnt}, loss: {loss}", f'accuracy:{100*(correct/size)}')
            summary_writer.add_scalar("accuracy", 100*(correct/size), global_step)    
            loss.backward()
            nn.utils.clip_grad_value_(model_tag.parameters(), 5.0)
            optimizer.step()

        model_tag.eval()
        correct = 0
        loss_sum = 0
        cnt = 0
        ep_cnt = 0
        with torch.no_grad():
            for i, sentence in tqdm(enumerate(val_data), total=len(val_data), desc=f"Epoch: {epoch}", unit="batches"):

                sentence_tag = sentence[:, :-1, 0].to(device)
                y_tag = sentence[:, -1, 0].to(device)

                y_pred_tag = model_tag(sentence_tag)

                correct += (y_pred_tag.argmax(dim=1) == y_tag).sum().item()

                loss = loss_function(y_pred_tag, y_tag.long())
                summary_writer.add_scalar("val_loss", loss, global_step)
                loss_sum+=loss

                ep_cnt += 1
                cnt += y_tag.size(0)
            print(f"Validation : loss {loss_sum/ep_cnt}, accuracy:{100* correct/cnt}")
        print(f"Epoch ended, time taken {time.time()-start_time}s")

    torch.save(model_tag, "D://data//first_model_tag1.pickle")


if __name__ == "__main__":
    train()
