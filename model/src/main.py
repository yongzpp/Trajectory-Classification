import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from net.config import cfg
from net.preprocess import *
from net.functions_class import SeqClassifier
import database.database_functions as db_func
from synthetic.create_dataset import create_dataset

import time

if __name__ == '__main__':
    if cfg.data.use_synthetic:
        try:
            trackpoints = db_func.read_synthetic_data()
            print("Synthetic data has already been created.")
        except Exception:
            cd = create_dataset()
            print("Synthetic dataset is now created.")
            trackpoints = db_func.read_synthetic_data()
    else:
        trackpoints = db_func.read_data()
    print("Data Loaded!")

    flights_df = format(trackpoints)
    df_train, df_test = train_test_split(flights_df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_train, test_size=0.3, random_state=1)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("Preparing for Training...")
    model = SeqClassifier()
    model.train(df_train, df_val)
    print("Model Trained!")

    print("Preparing for Testing...")
    model.test(df_test, cfg.model.dir_path+cfg.model.model_directory)
    print("Preparing for Prediction...")
    pred_df = model.predict(df_test.drop(columns=["Patterns"]), cfg.model.dir_path+cfg.model.model_directory)
    print("Labels Predicted!")

    print("Preparing to Export...")
    df_test = prepare_export(pred_df)
    df_test.to_csv("Test_Data.csv")
    print("CSV File Generated!")
    db_func.write(df_test, "Test_Data")
    print("Database Updated!")