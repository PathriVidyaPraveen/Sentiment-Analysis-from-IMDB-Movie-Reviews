import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongConcatAttention(nn.Module):
    def __init__(self,hidden_dim,attention_dim):
        super(LuongConcatAttention,self).__init__()
        self.linear = nn.Linear(hidden_dim*2 , attention_dim)
        self.v = nn.Linear(attention_dim,1,bias=False)

    def forward(self,encoder_outputs,decoder_hidden,mask=None):

        batch_size,seq_len,hidden_dim = encoder_outputs.size()
        decoder_expanded = decoder_hidden.unsqueeze(1).repeat(1,seq_len,1)
        concat_input = torch.cat((encoder_outputs , decoder_expanded) , dim = 2)
        energy = torch.tanh(self.linear(concat_input))
        attention_scores = self.v(energy).squeeze(2)

        if mask is not None:
            attention_scores = attention_scores.masked_fill( mask == 0 , -1e10)

        attention_weights = F.softmax(attention_scores , dim =1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context , attention_weights
    

