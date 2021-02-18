import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
import scipy.interpolate
from annotationapp import config as cfg


def fill_track_na_altitude(track):
    track = track.copy()
    track = track.reset_index(drop=True)
    if np.isnan(track.loc[0,'altitude']):
        i = 1
        while np.isnan(track.loc[i,'altitude']):
            i+=1
        track.loc[:i-1,'altitude'] = track.loc[i,'altitude']
        for j in range(i+1,len(track)):
            if np.isnan(track.altitude.loc[j]):
                track.loc[j,'altitude'] = track.altitude.loc[j-1] 
        
    else:
        for i in range(1,len(track)):
            if np.isnan(track.altitude.loc[i]):
                track.loc[i,'altitude'] = track.altitude.loc[i-1]
    return track

def get_t(combined):
    time_combined = combined.datetime
    start_time = time_combined.min()
    try:
        combined.loc[:,'t'] = (combined.datetime-start_time).dt.total_seconds()
    except Exception:
        combined.loc[:,'t'] = combined.datetime-start_time
    return combined

def fit_f1(x, a, b):
    return b*x + a

def fit_f5(x,a,b,c,d,e,f):
    
    return f*(x**5)+e*(x**4)+d*(x**3)+c*(x**2)+b*x+a
def fit_f7(x,a,b,c,d,e,f,g,h):
    return h*(x**7)+g*(x**6)+f*(x**5)+e*(x**4)+d*(x**3)+c*(x**2)+b*x+a

def fit_f9(x,a,b,c,d,e,f,g,h,i,j):
    return j*(x**9)+i*(x**8)+h*(x**7)+g*(x**6)+f*(x**5)+e*(x**4)+d*(x**3)+c*(x**2)+b*x+a
def get_smoothen_track(df,fit_f=fit_f1):
    df = get_t(df)
    for cord in ['latitude','longitude']:
        popt, pcov = curve_fit(fit_f, df['t'].values, df[cord].values)
        df['{}_smooth'.format(cord)] = df.t.apply(lambda x: fit_f(x,*popt))
    return df

def _interpolate_cord(df,cord):
    bw =3
    return scipy.interpolate.interp1d(df['t'].values,df[cord].values,bw)#(df['t'].values)

def _interpolate_df(df):
    models = {}
    for cord in ['latitude','longitude','altitude']:
        models[cord] = _interpolate_cord(df,cord)
    return models

def get_short_time(true_start,t1,t2):
    t1_sec = int((t1-true_start).total_seconds()+1)
    t2_sec = int((t2-true_start).total_seconds()-1)
    return t1_sec,t2_sec
def get_interpolation(df):
    df = df.reset_index(drop=True).copy()
    df = get_t(df)
    models = _interpolate_df(df)
    gaps = df.datetime.diff().dt.total_seconds()
    df['big_gap'] = gaps>cfg.INTERPOLATION_TRIGGER_GAP
    added_df = {
        'altitude':[],
        'latitude':[],
        'longitude':[],
        'datetime': [],
    }

    for i, row in df.iterrows():
        if row.big_gap:
            time_start = df.iloc[i-1].datetime
            time_end = row.datetime
            ts_sec,te_sec = get_short_time(df.datetime.min(),time_start,time_end)
            t_col = range(ts_sec,te_sec+1,cfg.INTERPOLATION_GAP)
            for cord in ['altitude','longitude','latitude']:
                added_df[cord] = added_df[cord] + models[cord](t_col).tolist()
            added_df['datetime'] = added_df['datetime']+  [df.datetime.min()+ pd.Timedelta(t,unit='s') for t in t_col]
    return pd.DataFrame.from_dict(added_df)

    #flagged = this point is after a long time
    