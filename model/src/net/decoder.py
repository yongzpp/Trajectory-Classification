import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

class Decoder(pl.LightningModule):
    """
    Decoder Object
    """
    def __init__(self, output_dim, hid_dim, n_layers, dropout, attention):
        """
        Initialises the decoder object.
        :param output_dim: number of classes to predict
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param dropout: dropout ratio for decoder
        :param attention: attention object to used (initialized in seq2seq)
        """
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, hid_dim)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + hid_dim, output_dim)
        self.rnn = nn.LSTM(hid_dim * 2 + hid_dim, hid_dim, n_layers, dropout = dropout).float()

    def forward(self, input, hidden, encoder_outputs, mask):
        """
        Forward propagation.
        :param input: features of dataset at every timestamp (tensor)
        :param hidden: hidden state from previous timestamp (tensor)
        :param encoder_outputs: used to measure similiarty in states in attention
        :param mask: mask to filter out the paddings in attention object
        :return: normalized output probabilities for each timestamp - softmax (tensor)
        """
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        input = self.dropout(embedded)
        a = self.attention(hidden[0], encoder_outputs, mask)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((input, weighted), dim = 2)
        output, (hidden, cell) = self.rnn(rnn_input, hidden)

        embedded = input.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        return F.log_softmax(prediction), (hidden,cell)