import torch
import torch.nn as nn
import pytorch_lightning as pl

class Encoder(pl.LightningModule):
    """
    Encoder Object
    """
    def __init__(self, hid_dim, n_layers, n_features, dropout):
        """
        Initialises the encoder object.
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param n_features: number of features from the dataset
        :param dropout: dropout ratio for encoder
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_features = n_features
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim*2, hid_dim)
        self.rnn = nn.LSTM(n_features, hid_dim, n_layers, dropout = dropout, bidirectional = True).float()
        
    def forward(self, x, src_len):
        """
        Forward propagation.
        :param x: features of dataset at every timestamp (tensor)
        :param src_len: actual length of each data sequence
        :return: hidden state of the last layer in the encoder;
                 outputs the outputs of last layer at every timestamps
        """
        tmp = None
        tmp_cell = None
        x = self.dropout(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, src_len)
        packed_outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = hidden.view(self.n_layers, 2, hidden.shape[1], hidden.shape[2])
        cell = cell.view(self.n_layers, 2, cell.shape[1], cell.shape[2])
        for i in range(self.n_layers):
            hidden_fwd = hidden[i][0]
            hidden_back = hidden[i][1]
            cell_fwd = cell[i][0]
            cell_back = cell[i][1]
            hidden_tmp = torch.cat((hidden_fwd, hidden_back), dim = 1)
            hidden_tmp = torch.tanh(self.fc(hidden_tmp))
            cell_tmp = torch.cat((cell_fwd, cell_back), dim = 1)
            cell_tmp = torch.tanh(self.fc(cell_tmp))
            if tmp == None:
                hidden_tmp = hidden_tmp.unsqueeze(0)
                tmp = hidden_tmp
            else:
                hidden_tmp = hidden_tmp.unsqueeze(0)
                tmp = torch.cat((tmp, hidden_tmp), dim = 0)
            if tmp_cell == None:
                cell_tmp = cell_tmp.unsqueeze(0)
                tmp_cell = cell_tmp
            else:
                cell_tmp = cell_tmp.unsqueeze(0)
                tmp_cell = torch.cat((tmp_cell, cell_tmp), dim = 0)
        return (tmp, tmp_cell), outputs