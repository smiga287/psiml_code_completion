import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMTagger import LSTMTagger
from LSTMValue import LSTMValue
from DataManager import DataManager
from util import SMALL, TRAIN, DATA_ROOT
from Dataset import DatasetPointer
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PointerMixtureModel import PointerMixtureModel
import warnings


def create_values(vector):
    str_to_idx = {s: i for i, (_, s, _, _) in enumerate(vector)}
    x = [str_to_idx[val] for _, val, _, _ in vector]
    return torch.Tensor(x)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TAG_EMBEDDING_DIM = 64
    VAL_EMBEDDING_DIM = 128
    HIDDEN_DIM = 1500
    NUM_EPOCHS = 2
    LAYER_NUM = 1
    BATCH_SIZE = 256

    data_manager = DataManager(SMALL)
    warnings.filterwarnings("ignore")

    tag_to_idx, idx_to_tag = data_manager.get_tag_dicts()
    val_to_idx, idx_to_val = data_manager.get_val_dicts()

    train_split_idx = int(len(data_manager.get_data()) * 0.8)
    validate_split_idx = int(len(data_manager.get_data()) * 0.9)

    data_train = torch.Tensor(
        [
            (
                tag_to_idx[(tag, have_children, have_sibling)],
                val_to_idx.get(val, val_to_idx["UNK"]),
            )
            for tag, val, have_children, have_sibling in (
                data_manager.get_data()[:train_split_idx]
            )
        ]
    )
    values_train = create_values(data_manager.get_data()[:train_split_idx])

    data_eval = torch.Tensor(
        [
            (
                tag_to_idx[(tag, have_children, have_sibling)],
                val_to_idx.get(val, val_to_idx["UNK"]),
            )
            for tag, val, have_children, have_sibling in (
                data_manager.get_data()[train_split_idx:validate_split_idx]
            )
        ]
    )
    values_eval = create_values(
        data_manager.get_data()[train_split_idx:validate_split_idx]
    )

    train_data_loader = torch.utils.data.DataLoader(
        DatasetPointer(data_train, values_train),
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    eval_data_loader = torch.utils.data.DataLoader(
        DatasetPointer(data_eval, values_eval),
        BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=8,
    )

    model = nn.DataParallel(
        PointerMixtureModel(
            len(tag_to_idx),
            len(val_to_idx),
            TAG_EMBEDDING_DIM,
            VAL_EMBEDDING_DIM,
            HIDDEN_DIM,
            LAYER_NUM,
        )
    )
    # model = torch.load()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    # -----------putting model on GPU--------------
    model.to(device=device)
    # ---------------------------------------------

    model_iter = 1
    summary_writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):

        for i, (sentence, y) in tqdm(
            enumerate(train_data_loader),
            total=len(train_data_loader),
            desc=f"Epoch: {epoch}",
            unit="batches",
        ):
            global_step = epoch * len(train_data_loader) + i
            size = int(sentence.size(0))

            model.zero_grad()
            model.train()

            # **********************TODO******************debug training and change eval**************************************************************

            # creating mask (don't wont to predict batches where there is no right answer)
            unk_idx = val_to_idx["UNK"]
            mask_unk = y[:, 1] != unk_idx  # mask for all y val that are not UNK
            y_pt = torch.zeros(BATCH_SIZE, sentence.size(1)-1)
            mask_pt = torch.zeros(BATCH_SIZE).bool()
            for i, batch in enumerate(sentence[:,1:,2]):
                for j, tmp in enumerate(reversed(batch)):
                    if tmp == y[i, 2]:
                        y_pt[i,y_pt.size(1)-1-j] = 1
                        mask_pt[i]= 1
                        break
            y_at = F.one_hot(y[:,1].long())
            mask_at = y[:,1].long() == unk_idx
            y_at[mask_at,unk_idx] = 0
            mask = (mask_pt | mask_unk)>0

            sentence = sentence.to(device=device)
            st, y_pred_val, y_pred_pt = model(sentence[mask,:,:-1])
            y = y.to(device=device)

            loss = st*nn.BCELoss()(y_pred_val.float(),y_at[mask].float()) 
            loss+= (1-st)*nn.BCELoss()(y_pred_pt.float(), y_pt[mask].float())
            loss = loss.mean()
            correct=0
            mask_st = st>0
            mask_st.squeeze_(dim=1)
            y_tmp = y[mask]
            mask_correct = y[mask,1] != unk_idx
            mask_correct = mask_correct[mask_st]
            correct= (y_pred_val[mask_correct].argmax(dim=1) == y_tmp[mask_correct, 1]).sum().item()
        
            mask_correct = mask_pt[mask]>0
            mask_correct = mask_correct[mask_st]
            correct+= (y_pred_pt[mask_correct].argmax(dim=1) == (y_pt[mask])[mask_correct]).sum().item()

            summary_writer.add_scalar("model_pt: train loss", loss_tag, global_step)
            summary_writer.add_scalar(
                "model_pt: accuracy", 100 * (correct_tag / size), global_step
            )

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()

            if (i+1) % 200 == 0:
                val = f"val_pt accuracy: {100 * (correct / size)}, val_pt loss: {loss}\n"
                with open(f'{DATA_ROOT}log_pt.txt', 'a') as log:
                    log.write(val)
                
            TIME_FOR_EVAL = 2500
            if (i + 1) % TIME_FOR_EVAL == 0:
                #evaluation
                torch.save(model, f"{DATA_ROOT}models//pt//budala_{model_iter}.pickle")
                model_iter += 1
                model.eval()

                correct_sum = 0
                loss_sum = 0
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