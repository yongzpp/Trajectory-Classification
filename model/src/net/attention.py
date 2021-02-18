import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Attention(pl.LightningModule):
    '''
    Attention Object
    '''
    def __init__(self, hid_dim):
        """
        Initialises the attention object.
        :param hid_dim: hidden dimensions in each layer
        """
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        """
        Forward propagation.
        :param hidden: hidden state from previous timestamp (tensor)
        :param encoder_outputs: used to measure similiarty in states
        :return: normalized probabilities for each timestamp - softmax (tensor)
        """
        hidden = hidden[-1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)