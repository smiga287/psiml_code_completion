import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMTagger import LSTMTagger
from LSTMValue import LSTMValue
from DataManager import DataManager
from util import SMALL, TRAIN, TEST, DATA_ROOT
from Dataset import Dataset
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from AttentionModel import AtentionModel
import warnings


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TAG_EMBEDDING_DIM = 64
    VAL_EMBEDDING_DIM = 128
    HIDDEN_DIM = 1500
    NUM_EPOCHS = 2
    LAYER_NUM = 1
    BATCH_SIZE = 256

    data_manager_train = DataManager(TRAIN)
    data_manager_eval = DataManager(TEST)
    warnings.filterwarnings("ignore")

    tag_to_idx, idx_to_tag = data_manager_train.get_tag_dicts()
    val_to_idx, idx_to_val = data_manager_train.get_val_dicts()
    
    
    validate_split_idx = int(len(data_manager_eval.get_data()) * 0.04) # 2000 za eval


    data_train = torch.Tensor(
        [
            (
                tag_to_idx.get((tag, have_children, have_sibling), tag_to_idx["UNK"]),
                val_to_idx.get(val, val_to_idx["UNK"]),
            )
            for tag, val, have_children, have_sibling in (
                data_manager_train.get_data()
            )
        ]
    )

    data_eval = torch.Tensor(
        [
            (
                tag_to_idx.get((tag, have_children, have_sibling), tag_to_idx["UNK"]),
                val_to_idx.get(val, val_to_idx["UNK"]),
            )
            for tag, val, have_children, have_sibling in (
                data_manager_eval.get_data()[:validate_split_idx]
            )
        ]
    )

    train_data_loader = torch.utils.data.DataLoader(
        Dataset(data_train), BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8
    )

    eval_data_loader = torch.utils.data.DataLoader(
        Dataset(data_eval), BATCH_SIZE, shuffle=False, drop_last=True, num_workers=8
    )

    model_tag = nn.DataParallel(
        AtentionModel(
            len(tag_to_idx),
            len(val_to_idx),
            TAG_EMBEDDING_DIM,
            VAL_EMBEDDING_DIM,
            HIDDEN_DIM,
            LAYER_NUM,
            False
        )
    )

    model_val = nn.DataParallel(
        AtentionModel(
            len(tag_to_idx),
            len(val_to_idx),
            TAG_EMBEDDING_DIM,
            VAL_EMBEDDING_DIM,
            HIDDEN_DIM,
            LAYER_NUM,
            True
        )
    )

    #model = torch.load(f"D://data//model_attention_1.pickle")
    loss_function = nn.NLLLoss()
    optimizer_tag = optim.Adam(model_tag.parameters())
    optimizer_val = optim.Adam(model_val.parameters())

    # -----------putting models on GPU-------------
    model_tag.cuda()
    model_val.cuda()
    # ---------------------------------------------

    model_iter = 1

    # Sluzi za Tensorboard
    summary_writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):

        model_tag.train()
        model_val.train()

        for i, (sentence, y) in tqdm(
            enumerate(train_data_loader),
            total=len(train_data_loader),
            desc=f"Epoch: {epoch}",
            unit="batches",
        ):
            global_step = epoch * len(train_data_loader) + i
            size = int(sentence.size(0))

            model_tag.zero_grad()
            model_val.zero_grad()
            model_tag.train()
            model_val.train()

            unk_idx = val_to_idx["UNK"]
            mask_unk = y[:, 1] != unk_idx  # mask for all y val that are not UNK


            sentence_tag = sentence.to(device)
            y_pred_tag = model_tag(sentence_tag)
            y = y.to(device)
            
            correct_tag = (y_pred_tag.argmax(dim=1) == y[:, 0]).sum().item()
            loss_tag = loss_function(y_pred_tag, y[:, 0].long())

            summary_writer.add_scalar("model_tag: train loss", loss_tag, global_step)
            summary_writer.add_scalar(
                "model_tag: accuracy", 100 * (correct_tag / size), global_step
            )

            loss_tag.backward()
            nn.utils.clip_grad_value_(model_tag.parameters(), 5.0)
            optimizer_tag.step()

            loss_val = 0
            if mask_unk.sum() > 0:
                # do forward for val_model
                sentence_val = sentence[mask_unk, :, :].to(device)
                y_pred_val = model_val(sentence_val)
                y = y.to(device)

                correct_val = (y_pred_val.argmax(dim=1) == y[mask_unk, 1]).sum().item()
                loss_val = loss_function(y_pred_val, y[mask_unk, 1].long())

                summary_writer.add_scalar("model_value: train loss", loss_val, global_step)
                summary_writer.add_scalar(
                    "model_value: train accuracy", 100 * (correct_val / size), global_step
                )

                loss_val.backward()
                nn.utils.clip_grad_value_(model_val.parameters(), 5.0)
                optimizer_val.step()

            if (i+1) % 200 == 0:
                tag = f"TRAIN tag accuracy: {100 * (correct_tag / size)}, tag loss: {loss_tag}, "
                val = f"val accuracy: {100 * (correct_val / size)}, val loss: {loss_val}\n"

                with open(f'{DATA_ROOT}log.txt', 'a') as log:
                    log.write(tag)
                    log.write(val)
                
            TIME_FOR_EVAL = 2500
            if (i + 1) % TIME_FOR_EVAL == 0:
                #evaluation
                torch.save(model_tag, f"D://data//models//tag//budala_{model_iter}.pickle")
                torch.save(model_val, f"D://data//models//val//budala_{model_iter}.pickle")
                model_iter += 1

                model_tag.eval()
                model_val.eval()

                correct_sum_tag = 0
                correct_sum_val = 0
                loss_sum_tag = 0
                loss_sum_val = 0
                size_sum_eval=0

                with torch.no_grad():

                    for i_eval, (sentence_eval, y_eval) in tqdm(
                        enumerate(eval_data_loader),
                        total=len(eval_data_loader),
                        desc=f"Epoch eval: {global_step//TIME_FOR_EVAL}",
                        unit="batches",
                    ):
                        global_step_eval = (global_step//TIME_FOR_EVAL)*len(eval_data_loader) + i_eval
                        size_eval = int(sentence_eval.size(0))
                        size_sum_eval += size_eval
                        sentence_eval = sentence_eval.to(device)

                        unk_idx = val_to_idx["UNK"]
                        mask_unk = y_eval[:, 1] != unk_idx

                        #tag
                        sentence_tag = sentence_eval.to(device)
                        y_pred_tag = model_tag(sentence_tag)
                        y_eval = y_eval.to(device)
                        
                        correct_tag = (y_pred_tag.argmax(dim=1) == y_eval[:, 0]).sum().item()
                        loss_tag = loss_function(y_pred_tag, y_eval[:, 0].long())
                        
                        correct_sum_tag += correct_tag
                        loss_sum_tag += loss_tag

                        summary_writer.add_scalar("model_tag: evaluation loss", loss_tag, global_step_eval)
                        summary_writer.add_scalar(
                            "model_tag: evaluation accuracy", 100 * (correct_tag / size_eval), global_step_eval
                        )

                        if mask_unk.sum()>0:
                            sentence_eval = sentence_eval[mask_unk].to(device)
                            y_pred_val = model_val(sentence_eval)
                            y_eval = y_eval.to(device)

                            correct_val = (y_pred_val.argmax(dim=1) == y_eval[mask_unk, 1]).sum().item()
                            loss_val = loss_function(y_pred_val, y_eval[mask_unk, 1].long())

                            correct_sum_val += correct_val
                            loss_sum_val += loss_val
                            
                            summary_writer.add_scalar("model_value: evaluation loss", loss_val, global_step_eval)
                            summary_writer.add_scalar(
                                "model_value: evaluation accuracy", 100 * (correct_val / size_eval), global_step_eval
                            )

                    summary_writer.add_scalar("model_tag: average evaluation loss", loss_sum_tag/len(eval_data_loader), global_step//TIME_FOR_EVAL)
                    summary_writer.add_scalar(
                        "model_tag: average evaluation accuracy", 100 * (correct_sum_tag / size_sum_eval), global_step//TIME_FOR_EVAL
                    ) 

                    summary_writer.add_scalar("model_value: average evaluation loss", loss_sum_val/len(eval_data_loader), global_step//TIME_FOR_EVAL)
                    summary_writer.add_scalar(
                        "model_value: average evaluation accuracy", 100 * (correct_sum_val / size_sum_eval), global_step//TIME_FOR_EVAL
                    )

                    tag = f"EVAL: tag accuracy: {100 * (correct_sum_tag / size_sum_eval)}, tag loss: {loss_sum_tag/len(eval_data_loader)}, "
                    val = f"val accuracy: {100 * (correct_sum_val / size_sum_eval)}, val loss: {loss_sum_val/len(eval_data_loader)}\n"

                    with open(f'{DATA_ROOT}log.txt', 'a') as log:
                        log.write(tag)
                        log.write(val)


if __name__ == "__main__":
    train()
