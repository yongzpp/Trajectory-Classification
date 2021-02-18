import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

from .config import cfg
from .encoder import Encoder
from .decoder import Decoder
from .attention import Attention

def calc_accuracy(output, Y, mask):
    """
    Calculate the accuracy (point by point evaluation)
    :param output: output from the model (tensor)
    :param Y: ground truth given by dataset (tensor)
    :param mask: used to mask out the padding (tensor)
    :return: accuracy used for validation logs (float)
    """
    _ , max_indices = torch.max(output.data,1)
    max_indices = max_indices.view(mask.shape[1], mask.shape[0]).permute(1,0)
    Y = Y.view(mask.shape[1], mask.shape[0]).permute(1,0)
    max_indices = torch.masked_select(max_indices, mask)
    Y = torch.masked_select(Y, mask)
    train_acc = (max_indices == Y).sum().item()/max_indices.size()[0]
    return train_acc

def weighted_loss_function(trg, output, label_index, mask):
    """
    Calculate the weighted loss (point by point evaluation)
    :param trg: ground truth given by dataset (tensor)
    :param output: output from the model (tensor)
    :label_index: contains the true class each data belongs to;
                  used for calculating proportion hence weights (tensor)
    :param mask: used to mask out the padding (tensor)
    :return: loss needed for backpropagation and logging (float)
    """
    trg = trg[1:].permute(1,0,2)
    output = output[1:].permute(1,0,2)
    label_index = label_index[1:].squeeze(2).permute(1,0)
    label_index = torch.masked_select(label_index, mask)
    key, value = torch.unique(label_index, return_counts=True)
    value = torch.div(value, torch.sum(label_index)) 
    value = torch.add(torch.neg(value), 1)
    for i in range(key.shape[0]):
        label_index[label_index == key[i]] = value[i]
    mask = mask.unsqueeze(2).expand(trg.size())
    trg = torch.masked_select(trg, mask)
    output = torch.masked_select(output, mask)
    label_mask = (trg != 0)
    selected = torch.masked_select(output, label_mask)
    weighted = torch.mul(selected, label_index)
    loss = -torch.sum(weighted) / weighted.size()[0]
    return loss

class Seq2Seq(pl.LightningModule):
    """
    Seq2seq Model Object
    """
    def __init__(self, lr, hid_dim, n_layers, n_features, enc_dropout, dec_dropout, labels_dct):
        """
        Initialises the seq2seq model object.
        All hyparams are initialized in config.yaml
        :param lr: learning rate for trainer
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param n_features: number of features in dataset used
        :param enc_dropout: dropout ratio for encoder
        :param dec_dropout: dropout ratio for decoder
        :param labels_dct: dictionary containing encoding of classes
        """
        super().__init__()
        self.learning_rate = lr
        self.labels_dct = labels_dct
        self.n_class = len(labels_dct)
        self.attn = Attention(hid_dim).cuda()
        self.encoder = Encoder(hid_dim, n_layers, n_features, enc_dropout).cuda()
        self.decoder = Decoder(self.n_class, hid_dim, n_layers, dec_dropout, self.attn).cuda()
        self.save_hyperparameters()
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def create_mask(self, label_index):
        """
        Create mask to filter out the paddings.
        :param label_index: contains the true class each data belongs to
        :return: boolean where paddings = False and non-padding otherwise (tensor)
        """
        mask = (label_index[1:] != 0).squeeze(2).permute(1,0)
        return mask
    
    def forward(self, src, trg, src_len, label_index, teacher_forcing_ratio = cfg.model.teacher_forcing):
        """
        Forward propagation.
        :param src: features of dataset (tensor)
        :param trg: ground truth of dataset (tensor)
        :param src_len: actual length of each data sequence (nd.array)
        :param label_index: contains the true class each data belongs to
        :param teacher_forcing_ratio: probabilty of using truth as inputs for decoder
        :return: output the class for every timestamps (tensor)
        """
        trg = trg.squeeze(2)
        trg = trg.long()
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        features = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, features).cuda()
        hidden, outputs_ = self.encoder(src, src_len)
        
        input = trg[0,:]
        mask = self.create_mask(label_index)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, outputs_, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

    def configure_optimizers(self):
        '''
        Optimizer
        Adam and Learning Rate Decay used. 
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode = 'min',
            factor = 0.5,
            patience = 50,
            cooldown = 0,
            eps =  0,
            verbose = True
            )
        metric_to_track = 'val_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': metric_to_track
            }

    def training_step(self, batch, batch_idx):
        '''
        Pytorch Lightning Trainer (training)
        '''
        src = batch[0].permute(1,0,2)
        trg = batch[1].permute(1,0,2)
        label_index = batch[2].permute(1,0,2)
        src_len = batch[3]
        mask = self.create_mask(label_index)
        output = self(src, label_index, src_len, label_index)
        loss = weighted_loss_function(trg,output,label_index,mask)
        output = output[1:].view(-1, self.n_class)
        label_index = label_index[1:].reshape(-1)
        label_index = label_index.long()
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx):
        '''
        Pytorch Lightning Trainer (validation)
        '''
        src = batch[0].permute(1,0,2)
        trg = batch[1].permute(1,0,2)
        label_index = batch[2].permute(1,0,2)
        src_len = batch[3]
        mask = self.create_mask(label_index)
        output = self(src, label_index, src_len, label_index, 0)
        loss = weighted_loss_function(trg,output,label_index,mask)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        label_index = label_index[1:].reshape(-1)
        label_index = label_index.long()
        acc = calc_accuracy(output,label_index,mask)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}