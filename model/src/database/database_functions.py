import pandas as pd
from .app import db
from sqlalchemy import func
import time
import datetime
from net.config import cfg

def read(table, columns):
  qstring='''SELECT {} from {}
              '''.format(columns, table)
  SQL_Query = pd.read_sql_query(qstring, db.engine)
  df = pd.DataFrame(SQL_Query)
  return df

def write(df, name):
  df.to_sql(name, con=db.engine, chunksize = 1000, if_exists='replace', index=False)

def read_synthetic_data():
    trackpoints = read("TRACKPOINTS", '*')

    return trackpoints

def read_data():
    trackpoints = read("TRACKPOINT", '*')
    segments = read("SEGMENT", '*')
    tracks = read("TRACK", '*')
    
    trackpoints['Patterns'] = ['null'] * trackpoints.shape[0] #instantiate to be null first
    for index, row in segments.iterrows():
        curr_id = row['track_id']
        tracks_df = tracks[tracks['track_id'] == curr_id] #filter tracks with the same id. There should only be 1 track
        tasking = tracks_df['tasking'].tolist()[0]
        tasksubtype = tracks_df['tasksubtype'].tolist()[0]
        seg_profile = row['segment_profile']
        newlabel = tasking + '/' + tasksubtype + '/' + seg_profile
        curr_start_dt = str(row['start_datetime']) #idea is to convert the strings to datetime objects for comparing against the datetime of each trackpoint to determine if the trackpoint belongs to this segment.
        curr_end_dt = str(row['end_datetime'])
        start_dt = datetime.datetime.strptime(curr_start_dt, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(curr_end_dt, '%Y-%m-%d %H:%M:%S')

        sub_track_df = trackpoints[trackpoints['track_id'] == curr_id] #filter for all trackpoints with the same id
        for index1, row1 in sub_track_df.iterrows():
            if trackpoints.at[index1, 'Patterns'] != 'null': #this means that the trackpoint has already been assigned, do not waste time looking at it again
                continue
            curr_dt = str(row1['datetime'])
            dt = datetime.datetime.strptime(curr_dt, '%Y-%m-%d %H:%M:%S')
            if start_dt <= dt and dt <= end_dt: #if true, then this means that the trackpoint belongs to the segment
                trackpoints.at[index1, 'Patterns'] = newlabel
    
    trackpoints = trackpoints[trackpoints['Patterns'] != 'null'] #remove all trackpoints which are not assigned to a new label. These points are invalid as they are not in any segment.
    
    counts = {}
    uniquetrackids = trackpoints['track_id'].unique()
    for i in uniquetrackids:
        sub_df = trackpoints[trackpoints['track_id'] == i]
        counts[i] = sub_df.shape[0]

    tracksid = []
    for i in counts:
        if counts[i] >= cfg.data.min_num:
            tracksid.append(i)
            
    updated_tracks = tracks[tracks['track_id'].isin(tracksid)]
    updated_segments= segments[segments['track_id'].isin(tracksid)]
    trackpoints = trackpoints[trackpoints['track_id'].isin(tracksid)]

    uniquelabels = trackpoints['Patterns'].unique()
    nums = []
    for i in uniquelabels:
        sub_df = trackpoints[trackpoints['Patterns'] == i]
        num_unique_flights = len(sub_df['track_id'].unique())
        if num_unique_flights > cfg.data.min_flights and num_unique_flights < cfg.data.max_flights:
            nums.append(i)

    patterns_ls = []
    for i in nums:
        sub_df = trackpoints[trackpoints['Patterns'] == i]
        patterns_ls.append(i)

    new_trackpoints = pd.DataFrame()
    for i in patterns_ls:
        tmp_df = trackpoints[trackpoints['Patterns'] == i]
        new_trackpoints = pd.concat([new_trackpoints, tmp_df])
        new_trackpoints = new_trackpoints.reset_index(drop=True)

    return new_trackpoints
