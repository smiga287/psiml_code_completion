import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LSTMValue import LSTMValue
from DataManager import DataManager, SMALL
from torch.utils.data import DataLoader
from Dataset import Dataset


def train():
    data_manager = DataManager(SMALL)
    val_to_idx, idx_to_val = data_manager.get_val_dicts()
    filtered_data = [val for _, val, _, _ in data_manager.get_data()]
    data = torch.Tensor([(val_to_idx[val], 0) for val in filtered_data])

    split_idx = int(0.8 * len(data))
    train_set = data[:split_idx]
    eval_set = data[split_idx:]

    model = LSTMValue()

    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_func = nn.CrossEntropyLoss()

    num_of_epochs = 8
    batch_size = 128

    train_dataloader = DataLoader(Dataset(train_set))
    eval_dataloader = DataLoader(Dataset(eval_set))
    for i in len(BATCH_SIZE):
        pass


if __name__ == "__main__":
    train()

