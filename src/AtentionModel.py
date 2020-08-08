import torch
import torch.nn as nn
import torch.nn.functional as F

class AtentionModel(nn.Module):

    def __init__(self, vocab_size_tag, vocab_size_val, embedding_dim_tag, embedding_dim_val, hidden_dim , layer_cnt):
        super(AtentionModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.tag_embeddings = nn.Embedding(vocab_size_tag, embedding_dim_tag)
        
        self.val_embeddings = nn.Embedding(vocab_size_val, embedding_dim_val)
        
        self.embedding_dim = embedding_dim_tag + embedding_dim_val
        
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
        embeds_val =  self.val_embeddings(sentence[:,:,1].long())
        embeds_tag = self.tag_embeddings(sentence[:,:,0].long())
        embeds = torch.cat( (embeds_tag, embeds_val), 2)
        lstm_out, (last_h, _) = self.lstm(embeds)
        last_h = last_h.squeeze(dim=0)
        mt = lstm_out[:, :-1, :]
        At =  self.v( torch.tanh(self.wm(mt) + self.wh(last_h.unsqueeze(dim=1))) )

        alfa = F.softmax(At, dim=1)

        ct = torch.matmul(mt.permute(0,2,1), alfa)
        ct.squeeze_(dim=2)
        Gt = self.wg(torch.cat((last_h, ct),1))
        Gt = torch.tanh(Gt)

        val = self.final_val(Gt)
        val = F.log_softmax(val, dim=1)
        return val
