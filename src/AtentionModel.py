import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):

    def __init__(self,tag_model, vocab_size_tag, vocab_size_val, embedding_dim_tag, embedding_dim_val, hidden_dim , layer_cnt):
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
        
        self.wm = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.v = nn.Linear(hidden_dim, 1, bias=False)

        self.wg = nn.Linear(2*hidden_dim, hidden_dim, bias=False)
        self.final_tag = nn.Linear(hidden_dim, vocab_size_tag)
        self.final_val = nn.Linear(hidden_dim, vocab_size_val)


    def forward(self, sentence):
        embeds = torch.cat( (self.tag_embeddings(sentence[:,:,0]), self.val_embeddings(sentence[:,:,1])), 2)
        lstm_out, (last_h, last_c) = self.lstm(embeds)
        last_h.squeeze_()
        mt = lstm_out[:, :-1, :]
        At =  self.v( F.tanh(self.wm(mt) + self.wh(last_h.unsqueeze_(dim=1))) )

        alfa = F.softmax(At)

        ct = torch.matmul(mt,alfa).squeeze()

        Gt = self.wg(torch.cat((last_h, ct),1))
        Gt = F.tanh(Gt)

        tag = self.final_tag(Gt)
        val = self.final_val(Gt)
        tag = F.log_softmax(tag, dim=1)
        val = F.log_softmax(val, dim=1)
        return tag_scores
