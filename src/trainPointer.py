import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMTagger import LSTMTagger
from LSTMValue import LSTMValue
from DataManager import DataManager
from util import SMALL, TRAIN
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
    NUM_EPOCHS = 8
    LAYER_NUM = 1
    BATCH_SIZE = 256

    data_manager = DataManager(SMALL)
    warnings.filterwarnings("ignore")

    tag_to_idx, idx_to_tag = data_manager.get_tag_dicts()
    val_to_idx, idx_to_val = data_manager.get_val_dicts()

    train_split_idx = int(len(data_manager.get_data()) * 0.08)
    validate_split_idx = int(len(data_manager.get_data()) * 0.09)

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

    data_val = torch.Tensor(
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
    values_val = create_values(
        data_manager.get_data()[train_split_idx:validate_split_idx]
    )

    train_data_loader = torch.utils.data.DataLoader(
        DatasetPointer(data_train, values_train),
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    val_data_loader = torch.utils.data.DataLoader(
        DatasetPointer(data_val, values_val),
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
    model.cuda()
    # ---------------------------------------------

    model_iter = 1

    for epoch in range(NUM_EPOCHS):

        summary_writer = SummaryWriter()

        model.train()

        for i, (sentence, y) in tqdm(
            enumerate(train_data_loader),
            total=len(train_data_loader),
            desc=f"Epoch: {epoch}",
            unit="batches",
        ):
            global_step = epoch * len(train_data_loader) + i
            size = int(sentence.size(0))

            model.zero_grad()

            # unk_idx = val_to_idx["UNK"]
            # mask_unk = y[:, 1] != unk_idx  # mask for all y val that are not UNK
            # if mask_unk.sum() == 0:
            #    continue
            # sentence = sentence[mask_unk, :, :]

            sentence = sentence.to(device)
            y_pred_val, model_type = model(sentence)
            y = y.to(device)

            correct_val = (y_pred_val.argmax(dim=1) == y[:, 0]).sum().item()

            loss_val = loss_function(y_pred_val, y[:, 0].long())

            summary_writer.add_scalar("Tag train loss", loss_val, global_step)
            summary_writer.add_scalar(
                "tag accuracy", 100 * (correct_val / size), global_step
            )

            loss_val.backward()

            nn.utils.clip_grad_value_(model.parameters(), 5.0)

            optimizer.step()

            if i % 50 == 0:
                # torch.save(model, f"D://data//budala_advanced_{model_iter}.pickle")
                print(
                    f"Test tag accuracy: {100 * (correct_val / size)}, tag loss: {loss_val}"
                )
                model_iter += 1

        # validation
        model.eval()

        # for metrics
        correct_val = 0
        loss_sum_val = 0
        cnt = 0
        ep_cnt = 0

        with torch.no_grad():
            for i, (sentence, y) in tqdm(
                enumerate(val_data_loader),
                total=len(val_data_loader),
                desc=f"Epoch: {epoch}",
                unit="batches",
            ):
                global_step_val = epoch * len(val_data_loader) + i

                # unk_idx = val_to_idx["UNK"]
                # mask_unk = (y[:, 1] == unk_idx) == False  # all seq that are not UNK
                # if mask_unk.sum() == 0:
                #     continue

                # sentence = sentence[mask_unk][:][:]
                sentence = sentence.to(device)
                y_pred_val = model(sentence)
                y = y.to(device)

                correct_val += (y_pred_val.argmax(dim=1) == y[:, 0]).sum().item()

                # loss_tag = loss_function(y_pred_tag, y_tag.long())
                loss_val = loss_function(y_pred_val, y[:, 0].long())

                # summary_writer.add_scalar("validation_loss_tag", loss_tag, global_step_val)
                summary_writer.add_scalar(
                    "validation_loss_tag", loss_val, global_step_val
                )
                # # loss_sum_tag += loss_tag
                loss_sum_val += loss_val

                ep_cnt += 1
                cnt += y.size(0)

            # print(
            #     f"Validation tag: loss {loss_sum_tag/ep_cnt}, accuracy:{100*correct_tag/cnt}"
            # )
            print(
                f"Validation tag: loss {loss_sum_val/ep_cnt}, accuracy:{100*correct_val/cnt}"
            )

        torch.save(model, f"D://data//model_attention_{epoch}.pickle")
        # torch.save(model_val, "D://data//second_model_val.pickle")


if __name__ == "__main__":
    train()
