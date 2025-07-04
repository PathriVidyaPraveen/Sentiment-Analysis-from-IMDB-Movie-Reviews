import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self,embedding_matrix,hidden_dim,output_dim,padding_idx=0,dropout=0.3,freeze_embeddings=False):
        super(VanillaRNN,self).__init__()
        vocab_size , embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze=freeze_embeddings,padding_idx=padding_idx)
        self.rnn = nn.RNN(embedding_dim , hidden_dim , batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self , input_ids , lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded , lengths.cpu(),batch_first=True, enforce_sorted = False)
        packed_output , hidden = self.rnn(packed)
        last_hidden = hidden.squeeze(0)
        output = self.fc(self.dropout(last_hidden))
        return output
    


class VanillaLSTM(nn.Module):
    def __init__(self,embedding_matrix,hidden_dim,output_dim,padding_idx=0,dropout=0.3,freeze_embeddings=False):
        super(VanillaLSTM,self).__init__()
        vocab_size , embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze= freeze_embeddings,padding_idx = padding_idx)
        self.lstm = nn.LSTM(embedding_dim , hidden_dim ,batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim , output_dim)

    def forward(self,input_ids , lengths):

        embedded = self.embedding(input_ids)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embedded , lengths.cpu() , batch_first=True,enforce_sorted = False)
        packed_output , (hidden , _) = self.lstm(packed_embeddings)
        last_hidden = hidden.squeeze(0)
        output = self.fc(self.dropout(last_hidden))

        return output
    


class BidirectionalRNN(nn.Module):
    def __init__(self,embedding_matrix, hidden_dim , output_dim , padding_idx = 0 , dropout = 0.3 , freeze_embeddings=False):
        super(BidirectionalRNN, self).__init__()
        vocab_size , embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix , freeze = freeze_embeddings , padding_idx = padding_idx)
        self.bidirectionalrnn = nn.rnn(embedding_dim , hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2 , output_dim)

    

    def forward(self,input_ids,lengths):

        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded , lengths.cpu(), batch_first=True,enforce_sorted=False)
        packed_output , hidden = self.rnn(packed)
        hidden_cat = torch.cat((hidden[0] , hidden[1]) , dim=1)
        output = self.fc(self.dropout(hidden_cat))

        return output
    



