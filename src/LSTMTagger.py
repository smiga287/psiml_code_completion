import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim , vocab_size, tagset_size, layer_cnt):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.h_0 = torch.nn.Parameter(torch.randn(hidden_dim))
        self.c_0 = torch.nn.Parameter(torch.randn(embedding_dim))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=layer_cnt)
        

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds, (self.h_0, self.c_0) )
        tag_space = self.hidden2tag(lstm_out.view(sentence.size(0), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
