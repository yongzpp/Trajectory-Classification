import pandas as pd
from annotationapp.app import db
from .dbmodels import TRACK,TRACKPOINT
from sqlalchemy import func
import datetime
from copy import deepcopy
import time

def delete_entries_on_date(date_selected):
    qstring = '''SELECT id from TRACK WHERE date(datetime)="{}"'''.format(str(date_selected.date()))
    SQL_Query = pd.read_sql_query(qstring, db.engine)
    ids = pd.DataFrame(SQL_Query).id.astype(int)
    db.session.query(TRACKPOINT).filter(TRACKPOINT.track_id.in_(ids)).delete(synchronize_session=False)
    db.session.query(TRACK).filter(func.DATE(TRACK.datetime)==date_selected.date()).delete(synchronize_session=False)
    db.session.commit()


def add_track_to_database(datetime,component_length,external_id):
    none_entry = 'None'
    entry=TRACK(
        datetime=datetime.to_pydatetime(),
        component_length=int(component_length),
        external_id = external_id
        )
    db.session.add(entry)
    db.session.commit()
    return entry.id

def add_trackpoints_to_database(df,track_id):
    entries=[]
    for _,row in df.iterrows():
        entries.append(TRACKPOINT(
            track_id=track_id,
            latitude=row.latitude,
            longitude=row.longitude,
            altitude=row.altitude,
            speed = row.speed,
            bearing = row.bearing,
            datetime=row.datetime.to_pydatetime(),
            millisecond=int(row.millisecond),
            ))
    db.session.add_all(entries)
    db.session.commit()

def read_dates():
    qstring='''SELECT date(datetime) from TRACK
                '''
    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)
    return df

def read_labels():
    qstring='''SELECT label from TRACK
                '''
    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)
    return df

def read_callsign(date):
    qstring='''SELECT DISTINCT callsign from TRACK
                '''
    if date!='all':
        qstring=qstring+''' WHERE date(datetime)='{}'
        '''.format(date)
    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)
    return df

def read_squadcode(date):
    qstring='''SELECT DISTINCT squadcode from TRACK
                '''
    if date!='all':
        qstring=qstring+''' WHERE date(datetime)='{}'
        '''.format(date)
    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)
    return df

def read_filtered_tracks(date_filter):
    qstring='''SELECT * from TRACK
                '''
    filt=False
    if date_filter!='all':
        print(date_filter)
        qstring=qstring+''' WHERE date(datetime)='{}'
        '''.format(date_filter)
        filt=True

    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)
    #check if at least 1 segment is present. Else, the track is deemed invalid.
    idxs_dropped = []
    for index, row in df.iterrows():
        track_id = row['track_id']
        segments = read_segments(track_id)
        if segments.shape[0] == 0:
            idxs_dropped.append(index)
            
    df.drop(idxs_dropped, inplace = True)
    
    return df

def read_track(track_id):
    qstring='''SELECT * from TRACK where id = {} 
                '''.format(track_id)
    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)
    return df

#stores the current state of the df so that only processing once is enough
actual_df = None
actual_filtered_df = None

def read_by_labels():
#For the real data, the function concatenates the tasking, tasksubtype, and segment profile to form the new label. For synthetic data, it will simply read all of the points from the table.

    start_time = time.time()
    global actual_df

    #retrieves all information from the 3 tables as dataframes
    qstring='''SELECT * from SEGMENT
                '''

    SQL_Query = pd.read_sql_query(qstring, db.engine)
    segments = pd.DataFrame(SQL_Query)

    qstring='''SELECT * from TRACKPOINT
                '''

    SQL_Query = pd.read_sql_query(qstring, db.engine)
    trackpoints = pd.DataFrame(SQL_Query)

    qstring='''SELECT * from TRACK
                '''

    SQL_Query = pd.read_sql_query(qstring, db.engine)
    tracks = pd.DataFrame(SQL_Query)

    trackpoints['new_label'] = ['null'] * trackpoints.shape[0] #instantiate to be null first
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
            if trackpoints.at[index1, 'new_label'] != 'null': #this means that the trackpoint has already been assigned, do not waste time looking at it again
                continue
            curr_dt = str(row1['datetime'])
            dt = datetime.datetime.strptime(curr_dt, '%Y-%m-%d %H:%M:%S')
            if start_dt <= dt and dt <= end_dt: #if true, then this means that the trackpoint belongs to the segment
                trackpoints.at[index1, 'new_label'] = newlabel
    
    trackpoints = trackpoints[trackpoints['new_label'] != 'null'] #remove all trackpoints which are not assigned to a new label. These points are invalid as they are not in any segment.

    uniquelabels = trackpoints['new_label'].unique()
    count_dct = {}
    for i in uniquelabels: #this is to determine the number of flights in each label
        sub_df = trackpoints[trackpoints['new_label'] == i]
        num_tracks = len(sub_df['track_id'].unique())
        count_dct[i] = num_tracks 

    lst = list(count_dct.items())
    lst.sort(key = lambda x: x[1], reverse = True) #sort in descending order since we are only interested in the top few labels with the most number of flights
    #lst = lst[:4] #only want the top 4 flights
    #print(lst)
    filtered_labels = list(map(lambda x: x[0], lst))
    trackpoints = trackpoints[trackpoints['new_label'].isin(filtered_labels)] #only keep those trackpoints which have one of the desired labels

    '''
    trackpoints['num_flights'] = [0] * trackpoints.shape[0]
    keys = list(map(lambda x: x[0], lst))
    for index, row in trackpoints.iterrows():
        curr_label = row['new_label']
        if curr_label in keys:
            trackpoints.at[index, 'num_flights'] = count_dct[curr_label]
    trackpoints = trackpoints[trackpoints['num_flights'] > 0]
    trackpoints = trackpoints.sort_values(by = ['num_flights'], ascending = False)
    '''
    '''
    uniquetrackids = trackpoints['track_id'].unique()
    overall_df = pd.DataFrame()
    for i in uniquetrackids: #idea is to retrieve only the start, middle and end pt of each trajectory
        sub_df = trackpoints[trackpoints['track_id'] == i]
        sub_df = deepcopy(sub_df)
        sub_df = sub_df.sort_values(by = ['datetime'])
        sub_df = sub_df.reset_index(drop=True)
        indices = list(sub_df.index)
        middle_index = sub_df.shape[0] // 2
        first_idx = indices[0]
        middle_idx = indices[middle_index]
        last_idx = indices[-1]
        new_df = sub_df.iloc[[first_idx, middle_idx, last_idx], ]
        overall_df = pd.concat([overall_df, new_df])
    '''
    #df = overall_df
    actual_df = trackpoints
    #df = df.sort_values(by=['datetime'],ascending=[True])

    end_time = time.time()
    tot_time = datetime.timedelta(seconds = end_time - start_time)
    #print(tot_time)
    return actual_df

dfpred = None
filtered_dfpred = None

def read_by_labelspred():
#for retrieve all data from predictions
    global dfpred
    curr_df = read_flightpoints()
    dfpred = curr_df
    return dfpred

def read_unique_labels(label):
#Filter and return the rows of the dataframe with the desired label for real data
    '''
    @param: label: a string, a pattern
    '''

    global actual_df
    global actual_filtered_df
    curr_df = deepcopy(actual_df)
    
    #print(label)
    if label == 'all' or 'all' in label:
        curr_df = curr_df
    else:
        try:
            curr_df = curr_df[curr_df['new_label'].isin(label)]
        except Exception: #length 1
            curr_df = curr_df[curr_df['new_label'] == label]
    
    actual_filtered_df = curr_df
    return actual_filtered_df

def read_unique_labelspred(label):
#Filter and return the rows of the dataframe with the desired label for predictions
    '''
    @param: label: a string, a pattern
    '''
    global dfpred
    global filtered_dfpred
    curr_df = deepcopy(dfpred)
    if label == 'all' or 'all' in label:
        curr_df = curr_df
    else:
        try:
            curr_df = curr_df[curr_df['Patterns'].isin(label)]
        except Exception: #length 1
            curr_df = curr_df[curr_df['Patterns'] == label]
    filtered_dfpred = curr_df
    return filtered_dfpred

def read_unique_tracks(label):
#Filter for points from real data belongning to that label
    '''
    @param: label: a string, track id
    '''
    global actual_filtered_df
    curr_df = deepcopy(actual_filtered_df)
    label = [i.replace('(Actual)', '') for i in label] #remove since it is not an actual part of the label
    label = [i.replace('(Predicted)', '') for i in label]
    label = [int(i) for i in label]
    if 'all' in label or label == 'all':
        return curr_df
    else:
        curr_df = curr_df[curr_df['track_id'].isin(label)]
    return curr_df

def read_unique_trackspred(label):
#Filter for points from predictions belongning to that label
    '''
    @param: label: a string, trajectory id
    '''
    global filtered_dfpred
    curr_df = deepcopy(filtered_dfpred)
    label = [i.replace('(Actual)', '') for i in label]
    label = [i.replace('(Predicted)', '') for i in label]
    label = [int(i) for i in label]
    if 'all' in label or label == 'all':
        return curr_df
    else:
        curr_df = curr_df[curr_df['track_id'].isin(label)]
    return curr_df

def read_trackpoints(track_id):
#no need for now
    qstring='''SELECT * from TRACKPOINT WHERE track_id ={}
                '''.format(track_id)

    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)

    segments = read_segments(track_id)
    num_unique_pats = segments.shape[0]
    pats_lst = []
    for i in range(num_unique_pats):
        pats_lst.append(i)
    fin = []
    for index, row in df.iterrows():
        curr_dt = str(row['datetime'])
        dt = datetime.strptime(curr_dt, '%Y-%m-%d %H:%M:%S')
        for index1, row1 in segments.iterrows():
            curr_start_dt = str(row1['start_datetime'])
            curr_end_dt = str(row1['end_datetime'])
            start_dt = datetime.strptime(curr_start_dt, '%Y-%m-%d %H:%M:%S')
            end_dt = datetime.strptime(curr_end_dt, '%Y-%m-%d %H:%M:%S')
            if start_dt <= dt and dt <= end_dt:
                fin.append(pats_lst[index1])
                break
    df['source_track_id'] = fin
    return df

def read_segments(track_id):
#no need for now
    qstring='''SELECT * from SEGMENT WHERE track_id ={}
                '''.format(track_id)

    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)
    return df

    

def query_track(track_id):
    q = db.session.query(TRACK).filter(TRACK.id==track_id).first()
    return q

def read_flightpoints():
#retrieve data from predictions
    qstring='''SELECT * from Test_Data
                '''
    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)
    return df

def read_filtered_flightpoints(traj_id):
    qstring='''SELECT * from Test_Data where track_id = {}
                '''.format(traj_id)

    SQL_Query = pd.read_sql_query(qstring, db.engine)
    df = pd.DataFrame(SQL_Query)
    return df