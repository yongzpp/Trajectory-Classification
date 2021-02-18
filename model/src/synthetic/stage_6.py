#!/usr/bin/python
import pandas as pd
import numpy as np
from .create_config import cfg

class stage_6:
    def __init__(self, stage_5_csv):
        #Flow: 
        #1. Convert non-normalised dataset to points for database
        self.stage_5_csv = stage_5_csv
        self.stage_6_csv = None

    #Stage 6: Convert non-normalised dataset to points for database
    def stage_6(self):
        '''
        return: a dataframe where a row represents a point
        '''
        self.convert_to_pts()
        return self.stage_6_csv

    def convert_to_pts(self):
        '''
        final output: a dataframe where a row represents a point
        '''
        #aim is to convert each row (which contains for example, 40 points in a trajectory) into 40 rows in the new dataframe where a row is a point
        dataframe = self.stage_5_csv

        catids = []
        trajids = []
        ts = []
        lats = []
        lons = []
        alts = []
        heads = []
        spds = []
        ms = []
        st = []
        source = []
        
        for index, row in dataframe.iterrows(): 
            curr_pat = row['Patterns']
            curr_ts = row['Timestamps']
            curr_lats = row['Latitudes']
            curr_lons = row['Longitudes']
            curr_alts = row['Altitudes']
            curr_heads = row['Headings']
            curr_speeds = row['Speeds']

            catids.extend(curr_pat)
            trajidxes = [index] * len(curr_pat)
            trajids.extend(trajidxes)
            ts.extend(curr_ts)
            lats.extend(curr_lats)
            lons.extend(curr_lons)
            alts.extend(curr_alts)
            heads.extend(curr_heads)
            spds.extend(curr_speeds)
            ms.extend([0] * len(curr_pat))
            st.extend([0] * len(curr_pat))
            source.extend([0] * len(curr_pat))


        pts_df = pd.DataFrame()
        pts_df['Patterns'] = catids
        pts_df['track_id'] = trajids
        pts_df['datetime'] = ts
        pts_df['latitude'] = lats
        pts_df['longitude'] = lons
        pts_df['altitude'] = alts
        pts_df['bearing'] = heads
        pts_df['speed'] = spds
        pts_df['millisecond'] = ms
        pts_df['stitch_type'] = st
        pts_df['source_track_id'] = source
        self.stage_6_csv = pts_df
        