import math
import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import Seq2Seq
from .preprocess import *
from .config import cfg
from .dataset import Flight_Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

N_FEATURES = cfg.data.features
N_EPOCHS = cfg.model.epochs
HID_DIM = cfg.model.hidden_size
N_LAYERS = cfg.model.hidden_layers
LEARNING_RATE = cfg.model.lr
ENC_DROPOUT = cfg.model.enc_dropout
DEC_DROPOUT = cfg.model.dec_dropout
SEED = cfg.model.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_accuracy(output,Y,mask):
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
    return train_acc, max_indices, Y

def loss_function(trg, output, mask):
    """
    Calculate the loss (point by point evaluation)
    :param trg: ground truth given by dataset (tensor)
    :param output: output from the model (tensor)
    :param mask: used to mask out the padding (tensor)
    :return: loss needed for backpropagation and logging (float)
    """
    trg = trg[1:].permute(1,0,2)
    output = output[1:].permute(1,0,2)
    mask = mask.unsqueeze(2).expand(trg.size())
    trg = torch.masked_select(trg, mask)
    output = torch.masked_select(output, mask)
    label_mask = (trg != 0)
    selected = torch.masked_select(output, label_mask)
    loss = -torch.sum(selected) / selected.size()[0]
    return loss

def default_collate(batch):
    """
    Stack the tensors from dataloader and pad sequences in batch
    :param batch: batch from the torch dataloader
    :return: stacked input to the seq2seq model
    """
    batch.sort(key=lambda x: x[3], reverse=True)
    batch_feature = [t[0] for t in batch]
    batch_labels = [t[1] for t in batch]
    batch_index = [t[2] for t in batch]
    batch_length = [t[3] for t in batch]
    batch_id = [t[4] for t in batch]
    batch_pad_feature = torch.nn.utils.rnn.pad_sequence(batch_feature, batch_first=True)
    batch_pad_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)
    batch_pad_index = torch.nn.utils.rnn.pad_sequence(batch_index, batch_first=True)
    return [batch_pad_feature, batch_pad_labels, batch_pad_index, batch_length, batch_id]

class SeqClassifier:
    """
    SeqClassifier Object
    """
    def __init__(self):
        '''
        Initialises the SeqClassifier object.
        '''
        self.labels_dct = None
        self.n_class = None

    def create_mask(self, label_index):
        """
        Create mask to filter out the paddings.
        :param label_index: contains the true class each data belongs to
        :return: boolean where paddings = False and non-padding otherwise (tensor)
        """
        mask = (label_index[1:] != 0).squeeze().permute(1,0)
        return mask

    def preprocess(self, df, mode, df_val=None):
        """
        Preprocess from a dataframe to numpy arrays to be inputed to model
        :param df: dataframe of the dataset given
        :param mode: train or val or test; train is used to generate the encodings of labels
        :param df_val: validation set for the model; create one if not given
        :return: dataloader for the model
        """
        X_df = to_array(df)
        if mode == "train":
            X_val = to_array(df_val)
            self.labels_dct, y_train, y_val = get_class(df, df_val)
            self.n_class = len(self.labels_dct)
            train_dataset = Flight_Dataset(X_df, y_train, self.labels_dct, "train")
            valid_dataset = Flight_Dataset(X_val, y_val, self.labels_dct, "val")
            train_loader = DataLoader(train_dataset, collate_fn=default_collate, batch_size=cfg.model.batch_size, shuffle=True, num_workers=4)
            valid_loader = DataLoader(valid_dataset, collate_fn=default_collate, batch_size=cfg.model.batch_size, shuffle=False, num_workers=4)
            return train_loader, valid_loader

        elif mode == "val":
            y_test = clean_labels(df)
            test_dataset = Flight_Dataset(X_df, y_test, self.labels_dct, "val")
            test_loader = DataLoader(test_dataset, collate_fn=default_collate, batch_size=cfg.model.batch_size, shuffle=False, num_workers=4)
            return test_loader

        elif mode == "test":
            y_test = clean_test(df)
            test_dataset = Flight_Dataset(X_df, y_test, self.labels_dct, "test")
            test_loader = DataLoader(test_dataset, collate_fn=default_collate, batch_size=cfg.model.batch_size, shuffle=False, num_workers=4)
            return test_loader

    def train(self, df_train, df_val=None):
        """
        Training Phase
        :param df_train: dataframe of the training dataset
        :param df_val: validation set for the model; create one if not given
        :return: best model and logs saved in specified directory
        """
        if not isinstance(df_val, pd.DataFrame):
            df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=1)
        train_loader, valid_loader = self.preprocess(df_train, "train", df_val)

        with torch.no_grad():
            torch.cuda.empty_cache()
        model = Seq2Seq(LEARNING_RATE, HID_DIM, N_LAYERS, N_FEATURES, ENC_DROPOUT, DEC_DROPOUT, self.labels_dct)

        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.model.dir_path,
            filename = 'best_model',
            verbose=True,
            monitor='val_loss',
            mode='min'
            )
        lr_logging_callback = LearningRateMonitor(logging_interval='step')

        callbacks = [checkpoint_callback,lr_logging_callback]
        trainer = pl.Trainer(
            gpus=1,
            callbacks=callbacks,
            max_epochs=N_EPOCHS,
            default_root_dir = cfg.model.dir_path
        )

        if cfg.model.auto_lr:
            lr_finder = trainer.tuner.lr_find(model,train_loader,valid_loader)
            new_lr = lr_finder.suggestion()
            model.learning_rate = new_lr

        trainer.fit(model, train_loader, valid_loader)

    def test(self, df_test, version):
        """
        Testing Phase, prints the accuracy
        :param df_test: dataframe of the dataset to be tested
        :param version: version name of the model to be loaded via specified directory
        """
        with torch.no_grad():
            torch.cuda.empty_cache()

        epoch_acc = 0
        epoch_loss = 0
        trajectory_ls = []
        actual_ls = []
        model = Seq2Seq.load_from_checkpoint(version)
        self.labels_dct = model.labels_dct
        model.eval()

        test_loader = self.preprocess(df_test, "val")

        wrong_dct = {}
        correct_dct = {}
        overall_df = pd.DataFrame()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                src = batch[0].permute(1,0,2).to(device)
                trg = batch[1].permute(1,0,2).to(device)
                label_index = batch[2].permute(1,0,2).to(device)
                src_len = batch[3]
                mask = self.create_mask(label_index)
                output = model(src, label_index, src_len, label_index, 0)
                loss = loss_function(trg,output,mask)
                epoch_loss += loss.item()
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                label_index = label_index[1:].reshape(-1)
                label_index = label_index.long()
                acc, output_ls, Y = calc_accuracy(output,label_index,mask)
                epoch_acc += acc 

                for i in range(len(output_ls)):
                    if output_ls[i] != Y[i]:
                        if Y[i].item() in wrong_dct.keys():
                            wrong_dct[Y[i].item()] += 1
                        else:
                            wrong_dct[Y[i].item()] = 1
                    else:
                        if Y[i].item() in correct_dct.keys():
                            correct_dct[Y[i].item()] += 1
                        else:
                            correct_dct[Y[i].item()] = 1

                overall_dct = {}
                for i in wrong_dct.keys():
                    if i in correct_dct.keys():
                        overall_dct[i] = correct_dct[i]/(correct_dct[i]+wrong_dct[i])
                    else:
                        overall_dct[i] = 0
                for i in self.labels_dct.values():
                    if i not in overall_dct.keys():
                        overall_dct[i] = 0

        print(overall_dct)
        print(self.labels_dct)
        print(epoch_acc / len(test_loader))
        print(epoch_loss / len(test_loader))

    def predict(self, df_test, version):
        """
        Prediction
        :param df_test: dataframe of the dataset to be predicted
        :param version: version name of the model to be loaded via specified directory
        :return: dataframe of the predicted test dataset with predicted labels
        """
        with torch.no_grad():
            torch.cuda.empty_cache()

        trajectory_ls = []
        actual_ls = []
        model = Seq2Seq.load_from_checkpoint(version)
        self.labels_dct = model.labels_dct
        model.eval()

        test_loader = self.preprocess(df_test, "test")
        overall_df = pd.DataFrame()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                src = batch[0].permute(1,0,2).to(device)
                trg = batch[1].permute(1,0,2).to(device)
                label_index = batch[2].permute(1,0,2).to(device)
                src_len = batch[3]
                mask = self.create_mask(label_index)
                output = model(src, label_index, src_len, label_index, 0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                label_index = label_index[1:].reshape(-1)
                label_index = label_index.long()
                acc, output_ls, Y = calc_accuracy(output,label_index,mask)

                df = df_test.copy()
                test_df = df.set_index('TrajectoryId').loc[batch[4]].reset_index(inplace=False)

                output_ls = to_trajectory(output_ls, src_len)
                labels_ls = to_trajectory(Y, src_len)
                output_ls = to_routes(output_ls, self.labels_dct)
                test_df["Patterns"] = output_ls
                overall_df = pd.concat([overall_df, test_df])
        return overall_df