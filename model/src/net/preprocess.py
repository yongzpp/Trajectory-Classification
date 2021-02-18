import re
import numpy as np
import pandas as pd
import torch
import scipy
from copy import deepcopy
import time

from .config import cfg

MAX_ALT = cfg.data.max_alt
MAX_LAT = cfg.data.max_lat
MAX_LON = cfg.data.max_lon
MAX_HEAD = cfg.data.max_head
MAX_SPD = cfg.data.max_spd

def clean_labels(df):
    """
    Converting labels to required dimensions 
    :param df: dataframe of dataset
    :return: array of labels in correct dimensions
    """
    target_ls = []
    cleaned_ls = []
    labels = df["Patterns"]
    for k in range(len(labels)):
        
        tmp_row = labels[k]
        add_dim = []
        for j in tmp_row:
            add_dim.append([j])
        target_ls.append(add_dim)
        
    for i in target_ls:
        
        targets = []
        for j in i:
            targets.append([j[0]])
        cleaned_ls.append(targets)
        
        
    return cleaned_ls

def clean_test(df):
    """
    Converting aribitary test labels to required dimensions 
    :param df: dataframe of dataset
    :return: array of labels (useless numbers) in correct dimensions
    """
    target_ls = []
    cleaned_ls = []
    labels = df["Timestamps"]
    for k in range(len(labels)):
        
        tmp_row = labels[k]
        add_dim = []
        for j in tmp_row:
            add_dim.append([j])
        target_ls.append(add_dim)
        
        

    for i in target_ls:
        
        targets = []
        for j in i:
            targets.append([j[0]])
        cleaned_ls.append(targets)
        
       
    return cleaned_ls

def create_dct(ls):
    """
    Create dictionary of encodings using training dataset
    :param ls: ls of labels of train dataset
    :return: dictionary of encodings
    """
    labels_dct = {}
    labels_dct["pad"] = 0
    index = 1
    for i in ls:
        for j in i:
            if j[0] not in labels_dct:
                labels_dct[j[0]] = index
                index += 1
    dic_len = len(labels_dct.keys())
    labels_dct["start"] = index
    labels_dct["unknown"] = index+1
    return labels_dct

def get_class(df_train, df_test):
    '''
    Preprocess the labels and generate dictionary of encodings
    :param df_train: dataframe of train dataset
    :param df_test: dataframe of test dataset
    :return: dictionary of encodings, train and test labels in correct dimensions
    '''
    y_train = clean_labels(df_train)
    y_test = clean_labels(df_test)
    labels_dct = create_dct(y_train)
    return labels_dct, y_train, y_test

def to_array(df):
    """
    Filter the required features
    :param df: dataframe of dataset
    :return: numpy array of dataset in float
    """
    features = df.iloc[range(df.shape[0]), 0:11].to_numpy().tolist()
    return features

def generate_seq(ls):
    """
    Generate array into sequences wrt timestamps
    :param ls: ls of preprocessed features of dataset
    :return: array of sequences of features
    """
    feature_seq = []
    for i in range(min(len(ls[0]), cfg.data.max_len)):
        ts = ls[0][i]
        lat = ls[1][i]
        lon = ls[2][i]
        alt = ls[3][i]
        head = ls[4][i]
        spd = ls[5][i]
        feature_seq.append(np.asarray([ts, lat, lon, alt, head, spd]))
    return np.asarray(feature_seq)

def preprocess_x(ls):
    """
    Data preprocessing of features
    :param ls: ls of features of dataset
    :return: array of sequences of features; length of each sequence
    """
    ls = numeric_stability(ls)
    ls = normalise_values(ls)
    processed = generate_seq(ls)
    return processed.astype(np.float32), len(processed), ls[6]

def assign_train(ls, dct):
    """
    Converting actual labels into label index encoding 
    :param ls: list of inputs of actual labels
    :param dct: dictionary of labels to encodings
    :return: list of inputs in their label index encodings
    """
    new_labels = []
    new_labels.append(np.array([np.float32(dct["start"])]))
    for i in range(min(len(ls), cfg.data.max_len)):
        new_labels.append(np.array([np.float32(dct[ls[i][0]])]))
    return np.array(new_labels)

def assign_val(ls, dct):
    """
    Converting actual labels into label index encoding ;
    Presence of unknown if not found in training
    :param ls: list of inputs of actual labels
    :param dct: dictionary of labels to encodings
    :return: list of inputs in their label index encodings
    """
    new_test = []
    new_test.append(np.array([np.float32(dct["start"])]))
    for i in range(min(len(ls), cfg.data.max_len)):
        if ls[i][0] in dct.keys():
            new_test.append(np.array([np.float32(dct[ls[i][0]])]))
        else:
            new_test.append(np.array([np.float32(dct["unknown"])]))
    return np.array(new_test)

def assign_test(ls, dct):
    """
    Converting arbitary labels into label index encoding;
    Replace all with unknown (to be predicted)
    :param ls: list of inputs of actual labels
    :param dct: dictionary of labels to encodings
    :return: list of inputs in their label index encodings
    """
    new_test = []
    new_test.append(np.array([np.float32(dct["start"])]))
    for i in range(min(len(ls), cfg.data.max_len)):
        new_test.append(np.array([np.float32(dct["unknown"])]))
    return np.array(new_test)

def one_hot(ls, dct):
    """
    Converting label index into one-hot encodings 
    :param ls: list of inputs of label_indexes encoding
    :param dct: dictionary of encodings to labels
    :return: list of inputs in their one hot encodings
    """
    encoded_labels = []
    for i in ls:
        empty = np.zeros(shape=(1, len(dct)), dtype=np.float32)
        empty[0][int(np.asarray(i[0]))] = np.float32(1)
        encoded_labels.append(np.asarray(empty[0]))
    return np.asarray(encoded_labels)

def preprocess_y(ls, dct, mode):
    """
    Preprocessing labels of dataset
    :param ls: list of inputs of actual labels
    :param dct: dictionary of encodings to labels
    :param mode: train or test; test includes unknown class if not in training
    :return: list of inputs in their one hot encodings
    """
    if mode == "train":
        convert_labels = assign_train(ls, dct)
    if mode == 'val':
        convert_labels = assign_val(ls, dct)
    elif mode == "test":
        convert_labels = assign_test(ls, dct)
    encoded = one_hot(convert_labels, dct)
    return encoded, convert_labels

def to_trajectory(ls, src_len):
    """
    Converting one-hot encodings to encodings (label index)
    :param ls: list of outputs one-hot encoded
    :param src_len: array of length for each sequence
    :return: list of outputs in their encodings (label index)
    """
    final = []
    pointer = 0
    for j in src_len:
        tmp = []
        for i in range(j):
            tmp.append(ls[pointer+i].item())
        pointer += j
        final.append(tmp)
    return final

def to_routes(ls, dct):
    """
    Converting encodings to actual labels
    :param ls: list of outputs in encodings
    :param dct: dictionary of encodings to labels
    :return: list of outputs in their actual labels
    """
    reversed_dct = {v: k for k, v in dct.items()}
    routes = []
    for i in ls:
        tmp = []
        for j in i:
            tmp.append(reversed_dct[j])
        routes.append(tmp)
    return routes

fast_counter = 0
def calculate_tmps(index, tmps):
    """
    Calculate the current accumulated timestamp.
    :param index: current index of the point in the trajectory
    :param tmps: list of timestamp values
    :return: fast_counter, which is an integer
    """
    global fast_counter
    if index == 0:
        return fast_counter
    else:
        fast_counter += tmps[index]
        return fast_counter

def format(df):
    """
    Convert trackpoints to tracks with list of points
    :param df: dataframe of trackpoints
    :return: dataframe of tracks with list of points for each feature
    """
    global fast_counter
    ts = []
    lats = []
    lons = []
    alts = []
    heads = []
    spds = []
    pats = []
    traj_id = []
    times = []
    ms = []
    stitch = []
    source = []
    unique_trajs = df['track_id'].unique()
    for i in unique_trajs:
        sub_df = df.loc[df['track_id'] == i]
        curr_ts = []
        curr_lats = []
        curr_lons = []
        curr_alts = []
        curr_heads = []
        curr_spds = []
        curr_pats = []
        curr_ms = []
        curr_stitch = []
        curr_source = []
        tmp = None
        pointer = None
        
        curr_lats = list(sub_df['latitude'].values)
        curr_lons = list(sub_df['longitude'].values)
        curr_alts = list(sub_df['altitude'].values)
        curr_heads = list(sub_df['bearing'].values)
        curr_spds = list(sub_df['speed'].values)
        curr_pats = list(sub_df['Patterns'].values)
        curr_ms = list(sub_df['millisecond'].values)
        curr_stitch = list(sub_df['stitch_type'].values)
        curr_source = list(sub_df['source_track_id'].values)
        if cfg.data.use_synthetic:
            curr_ts = list(sub_df['datetime'].values)
        else:
            curr_ts.append(0)
            pointer = sub_df.iloc[0, ]['datetime'].timestamp()
            tmp = 0

            new_df = deepcopy(sub_df)
            new_df['datetime'] = new_df['datetime'].apply(lambda row: row.timestamp())
            dtvalues = list(new_df['datetime'].values)[1:]
            pointers = list(new_df['datetime'].values)[:-1]
            dtvalues_np = np.array(dtvalues)
            pointers_np = np.array(pointers)
            diffs = dtvalues_np - pointers_np
            tmps = [tmp]
            tmps.extend(diffs)
            fast_counter = tmp
            tmps = [calculate_tmps(i, tmps) for i in range(len(tmps))]
            curr_ts.extend([diffs[i] + tmps[i] for i in range(len(diffs))])
            
        sub_df = sub_df.reset_index(drop=True)
        ts.append(curr_ts)
        lats.append(curr_lats)
        lons.append(curr_lons)
        alts.append(curr_alts)
        heads.append(curr_heads)
        spds.append(curr_spds)
        pats.append(curr_pats)
        traj_id.append(i)
        times.append(sub_df["datetime"][0])
        ms.append(curr_ms)
        stitch.append(curr_stitch)
        source.append(curr_source)
    
    routes_df = pd.DataFrame()
    routes_df['Timestamps'] = ts
    routes_df['Latitudes'] = lats
    routes_df['Longitudes'] = lons
    routes_df['Altitudes'] = alts
    routes_df['Headings'] = heads
    routes_df['Speeds'] = spds
    routes_df["TrajectoryId"] = traj_id
    routes_df["Datetime"] = times
    routes_df["Millisecond"] = ms
    routes_df["Stitch_Type"] = stitch
    routes_df["Source_Track_Id"] = source
    routes_df['Patterns'] = pats
    
    return routes_df

def numeric_stability(ls):
    """
    Ensure numerical stability by limiting upper bound
    :param ls: ls of trajectories
    :return: array of altered features
    """
    ts_ls = []
    alts_ls = []
    head_ls = []
    speeds_ls = []
    MAX_TS = max(ls[0])

    ts_ls = [ls[0][i] if ls[0][i] <= MAX_TS else MAX_TS for i in range(len(ls[0]))] 
    alts_ls = [ls[3][i] if ls[3][i] <= MAX_ALT else MAX_ALT for i in range(len(ls[3]))]
    head_ls = [ls[4][i] if ls[4][i] <= MAX_HEAD else MAX_HEAD for i in range(len(ls[4]))]
    speeds_ls = [ls[5][i] if ls[5][i] <= MAX_SPD else MAX_SPD for i in range(len(ls[5]))]
    
    
    return [ts_ls, ls[1], ls[2], alts_ls, head_ls, speeds_ls, ls[6]]

def normalise_values(ls):
    """
    Normalization of values
    :param ls: ls of trajectories
    :return: array of normalized features
    """
    ts_ls = []
    lats_ls = []
    longs_ls = []
    alts_ls = []
    head_ls = []
    speeds_ls = []
    MAX_TS = max(ls[0])
    
    ts_ls = [(ls[0][i] * 2 - MAX_TS) / MAX_TS for i in range(len(ls[0]))]
    lats_ls = [ls[1][i] / MAX_LAT for i in range(len(ls[1]))]
    longs_ls = [ls[2][i] / MAX_LON for i in range(len(ls[2]))]
    alts_ls = [(ls[3][i] * 2 - MAX_ALT) / MAX_ALT for i in range(len(ls[3]))]
    head_ls = [(ls[4][i] * 2 - MAX_HEAD) / MAX_HEAD for i in range(len(ls[4]))]
    speeds_ls = [(ls[5][i] * 2 - MAX_SPD) / MAX_SPD for i in range(len(ls[5]))]

    return [ts_ls, lats_ls, longs_ls, alts_ls, head_ls, speeds_ls, ls[6]]

def prepare_export(df):
    """
    Convert the dataframe of trajectories to trackpoints
    :param df: dataframe of trajectories
    :return: dataframe of trackpoints
    """
    overall_ts = []
    overall_lat = []
    overall_lon = []
    overall_alt = []
    overall_head = []
    overall_spd = []
    overall_pat = []
    overall_traj = []
    overall_ms = []
    overall_stitch = []
    overall_source = []
    for index, row in df.iterrows(): 
        curr_ts = row['Timestamps']
        curr_lat = row['Latitudes']
        curr_lon = row['Longitudes']
        curr_alt = row['Altitudes']
        curr_head = row['Headings']
        curr_spd = row['Speeds']
        curr_pat = row['Patterns']
        curr_traj = row["TrajectoryId"]
        curr_ms = row["Millisecond"]
        curr_stitch = row["Stitch_Type"]
        curr_source = row["Source_Track_Id"]
        #curr_time = row["Datetime"].timestamp()
        curr_time = pd.to_datetime(row["Datetime"], unit = 's').timestamp()
        for i in range(len(curr_ts)):
            curr_time += curr_ts[i]
            overall_ts.append(pd.to_datetime(curr_time, unit='s'))
            overall_lat.append(curr_lat[i])
            overall_lon.append(curr_lon[i])
            overall_alt.append(curr_alt[i])
            overall_head.append(curr_head[i])
            overall_spd.append(curr_spd[i])
            overall_ms.append(curr_ms[i])
            overall_stitch.append(curr_stitch[i])
            overall_source.append(curr_source[i])
            if i >= cfg.data.max_len:
                overall_pat.append(curr_pat[-1])
            else:
                overall_pat.append(curr_pat[i])
            overall_traj.append(curr_traj)
    
    df_test = pd.DataFrame(list(zip(overall_traj, overall_lat, overall_lon, overall_alt,
                                    overall_spd, overall_head, overall_ts, overall_ms,
                                    overall_stitch, overall_source, overall_pat)), 
               columns =['track_id', 'latitude', 'longitude', 'altitude', 'speed',
                           'bearing', 'datetime', 'millisecond', 'stitch_type', 
                           'source_track_id', 'Patterns']) 
    return df_test