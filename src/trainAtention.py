import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMTagger import LSTMTagger
from LSTMValue import LSTMValue
from DataManager import DataManager, SMALL, TRAIN
from Dataset import Dataset
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from AtentionModel import AtentionModel

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TAG_EMBEDDING_DIM = 100
    VAL_EMBEDDING_DIM = 64
    HIDDEN_DIM = 1500
    NUM_EPOCHS = 2  # 8
    LAYER_NUM = 1
    BATCH_SIZE = 256

    data_manager = DataManager(TRAIN)

    d = data_manager.get_data()

    tag_to_idx, idx_to_tag = data_manager.get_tag_dicts()
    val_to_idx, idx_to_val = data_manager.get_val_dicts()

    # ad hoc adding of UNKOWN
    val_to_idx['UNK'] = len(val_to_idx)
    idx_to_val[len(val_to_idx) - 1] = 'UNK'

    train_split_idx = int(len(data_manager.get_data()) * 0.9)
    validate_split_idx = int(len(data_manager.get_data()))
    data_train = torch.Tensor(
        [
            (tag_to_idx[(tag, have_children, have_sibling)], val_to_idx.get(val, val_to_idx['UNK']))
            for tag, val, have_children, have_sibling in (
                data_manager.get_data()[:train_split_idx]
            )
        ]
    )
    data_val = torch.Tensor(
        [
            (tag_to_idx[(tag, have_children, have_sibling)], val_to_idx.get(val, val_to_idx['UNK']))
            for tag, val, have_children, have_sibling in (
                data_manager.get_data()[train_split_idx:validate_split_idx]
            )
        ]
    )

    training_data = torch.utils.data.DataLoader(
        Dataset(data_train), BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0
    )

    #test_data = None

    val_data = torch.utils.data.DataLoader(
        Dataset(data_val), BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0
    )

    model = AtentionModel(
        len(tag_to_idx), len(val_to_idx),TAG_EMBEDDING_DIM, VAL_EMBEDDING_DIM,HIDDEN_DIM, LAYER_NUM
    )
    
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    

    # -----------putting everything on GPU---------
    model.cuda()
    # model_val.cuda()
    # ---------------------------------------------


    model_iter = 1

    for epoch in range(NUM_EPOCHS):

        summary_writer = SummaryWriter()

        model.train()
        
        start_time = time.time()
        cnt = 0
        for i, (sentence, y) in tqdm(
            enumerate(training_data),
            total=len(training_data),
            desc=f"Epoch: {epoch}",
            unit="batches",
        ):
            global_step = epoch * len(training_data) + i
            size = int(sentence.size(0))

            model.zero_grad()

            sentence.to(device)

            y_pred_val = model(sentence)

            #correct_tag = (y_pred_tag.argmax(dim=1) == y_tag).sum().item()
            correct_val = (y_pred_val.argmax(dim=1) == y[:,1]).sum().item()

            #loss_tag = loss_function(y_pred_tag, y_tag.long())
            loss_val = loss_function(y_pred_val, y[:,1].long())

            # summary_writer.add_scalar("Tag train loss", loss_tag, global_step)
            # summary_writer.add_scalar(
            #     "Tag accuracy", 100 * (correct_tag / size), global_step
            # )
            summary_writer.add_scalar("Val train loss", loss_val, global_step)
            summary_writer.add_scalar(
                # "Val accuracy", 100 * (correct_val / size), global_step
            )

            # loss_tag.backward()
            loss_val.backward()

            nn.utils.clip_grad_value_(model.parameters(), 5.0)
            # nn.utils.clip_grad_value_(model_val.parameters(), 5.0)

            optimizer.step()
            # optimizer_val.step()

            if i % 5000 == 0:
                torch.save(model_tag, f"D://data//budala_advanced_{model_iter}.pickle")
                model_iter += 1
                
        model_tag.eval()
        # model_val.eval()

        correct_tag = 0
        correct_val = 0

        loss_sum_tag = 0
        loss_sum_val = 0

        cnt = 0

        ep_cnt = 0
        with torch.no_grad():
            for i, (sentence,y) in tqdm(
                enumerate(val_data),
                total=len(val_data),
                desc=f"Epoch: {epoch}",
                unit="batches",
            ):
                global_step_val = epoch * len(val_data) + i
                
                sentence_tag = sentence[:, :, 0].to(device)
                y_tag = y[:, 0].to(device)
                y_pred_tag = model_tag(sentence_tag)

                # sentence_val = sentence[:, :, 1].to(device)
                # y_val = y[:, 1].to(device)
                # y_pred_val = model_val(sentence_val)

                correct_tag += (y_pred_tag.argmax(dim=1) == y_tag).sum().item()
                # correct_val += (y_pred_val.argmax(dim=1) == y_val).sum().item()

                loss_tag = loss_function(y_pred_tag, y_tag.long())
                # loss_val = loss_function(y_pred_val, y_val.long())

                summary_writer.add_scalar("validation_loss_tag", loss_tag, global_step_val)
                # summary_writer.add_scalar("validation_loss_val", loss_val, global_step_val)
                loss_sum_tag += loss_tag
                # loss_sum_val += loss_val

                ep_cnt += 1
                cnt += y_tag.size(0)

            print(
                f"Validation tag: loss {loss_sum_tag/ep_cnt}, accuracy:{100*correct_tag/cnt}"
            )
            # print(
            #     f"Validation val: loss {loss_sum_val/ep_cnt}, accuracy:{100*correct_val/cnt}"
            # )
        print(f"Epoch ended, time taken {time.time()-start_time}s")

    torch.save(model_tag, "D://data//first_model_tag.pickle")
    # torch.save(model_val, "D://data//second_model_val.pickle")


def validate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TAG_EMBEDDING_DIM = 64
    VAL_EMBEDDING_DIM = 64
    HIDDEN_DIM = 1500
    NUM_EPOCHS = 2  # 8
    LAYER_NUM = 1
    BATCH_SIZE = 256

    data_manager = DataManager(TRAIN)

    tag_to_idx, idx_to_tag = data_manager.get_tag_dicts()
    val_to_idx, idx_to_val = data_manager.get_val_dicts()

    # ad hoc adding of UNKOWN
    val_to_idx['UNK'] = len(val_to_idx)
    idx_to_val[len(val_to_idx) - 1] = 'UNK'

    train_split_idx = int(len(data_manager.get_data()) * 0.05)
    validate_split_idx = int(len(data_manager.get_data()) * 0.07)

    data_val = torch.Tensor(
        [
            (tag_to_idx[(tag, have_children, have_sibling)], val_to_idx.get(val, val_to_idx['UNK']))
            for tag, val, have_children, have_sibling in (
                data_manager.get_data()[train_split_idx:validate_split_idx]
            )
        ]
    )


    val_data = torch.utils.data.DataLoader(
        Dataset(data_val), BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0, pin_memory=True
    )

    model_tag = torch.load('D://data//budala_16.pickle')

    # model_val = LSTMValue(
    #     VAL_EMBEDDING_DIM, HIDDEN_DIM, len(val_to_idx), len(val_to_idx), LAYER_NUM
    # )
    loss_function = nn.NLLLoss()
    optimizer_tag = optim.Adam(model_tag.parameters())
    # optimizer_val = optim.Adam(model_val.parameters())

    # -----------putting everything on GPU---------
    model_tag.cuda()

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        summary_writer = SummaryWriter()

        model_tag.eval()
        # model_val.eval()
        correct_tag = 0
        # correct_val = 0

        loss_sum_tag = 0
        # loss_sum_val = 0

        cnt = 0

        ep_cnt = 0
        with torch.no_grad():
            for i, (sentence,y) in tqdm(
                enumerate(val_data),
                total=len(val_data),
                desc=f"Epoch: {epoch}",
                unit="batches",
            ):

                global_step_val = epoch * len(val_data) + i
                sentence_tag = sentence[:, :, 0].to(device)
                y_tag = y[:, 0].to(device)
                y_pred_tag = model_tag(sentence_tag)

                # sentence_val = sentence[:, :, 1].to(device)
                # y_val = y[:, 1].to(device)
                # y_pred_val = model_val(sentence_val)

                correct_tag += (y_pred_tag.argmax(dim=1) == y_tag).sum().item()
                # correct_val += (y_pred_val.argmax(dim=1) == y_val).sum().item()

                loss_tag = loss_function(y_pred_tag, y_tag.long())
                # loss_val = loss_function(y_pred_val, y_val.long())

                summary_writer.add_scalar("validation_loss_tag", loss_tag, global_step_val)
                
                loss_sum_tag += loss_tag
                

                ep_cnt += 1
                cnt += y_tag.size(0)

            print(
                f"Validation tag: loss {loss_sum_tag/ep_cnt}, accuracy:{100*correct_tag/cnt}"
            )
            # print(
            #     f"Validation val: loss {loss_sum_val/ep_cnt}, accuracy:{100*correct_val/cnt}"
            # )
        print(f"Epoch ended, time taken {time.time()-start_time}s")

if __name__ == "__main__":
    train()
