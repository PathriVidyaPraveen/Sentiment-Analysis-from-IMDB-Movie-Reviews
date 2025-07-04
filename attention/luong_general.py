import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongGeneralAttention(nn.Module):
    def __init__(self,hidden_dim):
        super(LuongGeneralAttention,self).__init__()
        self.W = nn.Linear(hidden_dim,hidden_dim,bias=False)
    

    def forward(self,encoder_outputs,decoder_hidden,mask=None):
        decoder_transformed = self.W(decoder_hidden)
        decoder_expanded = decoder_transformed.unsqueeze(1)
        attention_scores = torch.bmm(encoder_outputs, decoder_expanded.transpose(1,2)).squeeze(2)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0 , -1e10)
        attention_weights = F.softmax(attention_scores,dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1) , encoder_outputs).squeeze(1)

        return context , attention_weights
    


