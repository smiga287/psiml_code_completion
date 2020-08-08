import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):

    def __init__(self,tag_model, vocab_size_tag, vocab_size_val, embedding_dim_tag, embedding_dim_val, hidden_dim , vocab_size, tagset_size, layer_cnt):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.tag_embeddings = tag_model.word_embeddings
        for param in self.tag_embeddings.parameters():
            param.requires_grad = False
        
        self.val_embeddings = nn.Embedding(vocab_size_val, embedding_dim_val)
        
        self.embedding_dim = self.embedding_dim_tag + self.embedding_dim_val
        
        #***TODO****
        #self.h_0 = torch.nn.Parameter(torch.randn(hidden_dim))
        #self.c_0 = torch.nn.Parameter(torch.randn(embedding_dim))

        self.lstm = nn.LSTM( self.embedding_dim, hidden_dim, batch_first=True, num_layers=layer_cnt)
        
        self.wm = nn.Linear(hidden_dim, hidden_dim)
        self.wh = nn.Linear(hidden_dim, hidden_dim)

        self.v = nn.Linear(hidden_dim, 1)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = torch.cat( (self.tag_embeddings(sentence[:,:,0]), self.val_embeddings(sentence[:,:,1])), 2)
        lstm_out, (last_h, last_c) = self.lstm(embeds)
        last_h.squeeze_()
        mt = lstm_out[:, :-1, :]
        At =  self.v( self.wm(mt) + self.wh(last_h.unsqueeze_(dim=1)) )

        alfa = F.softmax(At)

        ct = torch.matmul((mt))

        tag_space = self.hidden2tag(mt)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
