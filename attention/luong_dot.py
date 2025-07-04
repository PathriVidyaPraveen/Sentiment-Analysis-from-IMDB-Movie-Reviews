import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongDotAttention(nn.Module):
    def __init__(self):
        super(LuongDotAttention,self).__init__()

    def forward(self,encoder_outputs , decoder_hidden , mask=None):
        decoder_hidden = decoder_hidden.unsqueeze(2)
        attention_scores = torch.bmm(encoder_outputs , decoder_hidden).squeeze(2)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0 , -1e10)
        attention_weights = F.softmax( attention_scores ,dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1) , encoder_outputs).squeeze(1)

        return context , attention_weights
    
