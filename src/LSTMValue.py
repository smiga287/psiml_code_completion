import torch.nn as nn
import torch.nn.functional as F


class LSTMValue(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, num_layers):
        super(LSTMValue, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence.long())
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out[:, -1, :])
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores
