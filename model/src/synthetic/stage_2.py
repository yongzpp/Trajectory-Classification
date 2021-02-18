#!/usr/bin/python
import pandas as pd
import numpy as np
from copy import deepcopy
from .create_config import cfg
import scipy.interpolate
import haversine as hs

class stage_2:
    def __init__(self, stage_1_csv):
        #Flow:
        #1. Add the threshold as start/end artificial points if the paths are too short
        #2. Interpolate to ensure that all routes in the same segment have the same number of points
        self.airports_lat_lon = cfg.preprocess.data.airports_lat_lon
        self.num_ts = cfg.preprocess.data.num_ts
        self.stage_1_csv = stage_1_csv
        self.stage_2_csv = None


    #Stage 2: Interpolate trajectories same segment same length 
    def stage_2(self):
        '''
        return: a dataframe where a row represents a trajectory
        '''
        self.make_all_routes_in_a_segment_same_length()
        return self.stage_2_csv

    def make_all_routes_in_a_segment_same_length(self):
        '''
        final output: a dataframe where a row represents a trajectory
        '''
        #same length meaning same number of points for all routes in a segment
        dataframe = self.stage_1_csv
        unique_segment_ids = dataframe['Segment Id'].unique()
        overall_update_df = pd.DataFrame()
        
        for i in unique_segment_ids:
            sub_df = dataframe[dataframe['Segment Id'] == i]
            update_sub_df = deepcopy(sub_df)

            #check start pt
            update_sub_df = self.add_start_pt(update_sub_df)

            #check end pt
            update_sub_df = self.add_end_pt(update_sub_df)

            #interpolate routes in segments
            update_sub_df = self.interpolate_routes_in_segment(update_sub_df)

            overall_update_df = pd.concat([overall_update_df, update_sub_df])

        class_labels = []
        for index, row in overall_update_df.iterrows():
            curr_segment = row['Segment Id']
            class_labels.append(str(curr_segment) + '-0') #the first segment will always be 0 at the end

        overall_update_df['Category'] = class_labels

        self.stage_2_csv = overall_update_df

    def add_start_pt(self, dataframe):
        '''
        @param: dataframe: 1 row represents a trajectory of points

        return: a dataframe where a row represents a trajectory
        '''
        #check start pt
        lats = []
        lons = []
        for index, row in dataframe.iterrows():
            lats.append(row['Latitudes'][0])
            lons.append(row['Longitudes'][0])

        airport_name = dataframe['Origin'].unique()[0]
        bot_lat = self.airports_lat_lon[airport_name][0]
        bot_lon = self.airports_lat_lon[airport_name][1]

        idxs_extrapolated = self.check_start_lats_lons(dataframe, [], bot_lat, lats)
        idxs_extrapolated = self.check_start_lats_lons(dataframe, idxs_extrapolated, bot_lon, lons)

        #add start point to those indices
        update_idxes = dataframe.index.tolist()
        for j in range(len(idxs_extrapolated)):
            curr_idx = update_idxes[j]
            curr_ts = self.stage_1_csv.iloc[curr_idx, ]['Timestamps']
            curr_lat = self.stage_1_csv.iloc[curr_idx, ]['Latitudes']
            curr_lon = self.stage_1_csv.iloc[curr_idx, ]['Longitudes']
            curr_alt = self.stage_1_csv.iloc[curr_idx, ]['Altitudes']
            curr_head = self.stage_1_csv.iloc[curr_idx, ]['Headings']
            curr_spd = self.stage_1_csv.iloc[curr_idx, ]['Speeds']
            curr_num_pts = self.stage_1_csv.iloc[curr_idx, ]['Number of points']
            lat = [bot_lat]
            lat.extend(curr_lat)
            lon = [bot_lon]
            lon.extend(curr_lon)
            new_ts = self.calculate_time(curr_ts, curr_lat, curr_lon, (bot_lat, bot_lon), 'start')
            ts = [new_ts]
            ts.extend(curr_ts)
            alt = [curr_alt[0]] #assume same alt, head, spd
            alt.extend(curr_alt)
            head = [curr_head[0]]
            head.extend(curr_head)
            spd = [curr_spd[0]]
            spd.extend(curr_spd)
            dataframe.at[curr_idx, 'Timestamps'] = ts
            dataframe.at[curr_idx, 'Latitudes'] = lat
            dataframe.at[curr_idx, 'Longitudes'] = lon
            dataframe.at[curr_idx, 'Altitudes'] = alt
            dataframe.at[curr_idx, 'Headings'] = head
            dataframe.at[curr_idx, 'Speeds'] = spd
            dataframe.at[curr_idx, 'Number of points'] += 1

        return dataframe

    def check_start_lats_lons(self, dataframe, idxs_extrapolated, thres, lats_lons):
        '''
        @param: dataframe: 1 row represents a trajectory of points.
                idxs_extrapolated: list of indices which have already been marked to add an arbitrary start point
                thres: threshold for latitude/longitude
                lats_lons: list of latitude/longitude values

        return: a list of indices where the trajectories have an artificial start point added
        '''
        #for those paths with starting lat/lon bigger than threshold, will add the threshold as the first point to the path. Only return indices
        update_idxes = dataframe.index.tolist()
        for j in range(len(lats_lons)):
            curr_lat = lats_lons[j]
            if curr_lat > thres + 0.01: #needa add min as start pt
                curr_idx = update_idxes[j]
                if curr_idx not in idxs_extrapolated:
                    idxs_extrapolated.append(curr_idx)

        return idxs_extrapolated

    def calculate_time(self, period, lats, lons, new_pt, start_or_end):
        '''
        @param: period: list of original timestamps in the route. 
                lats: list of original latitude values in the route.
                lons: list of original longitude values in the route.
                new_pt: tuple of latitude and longitude for the fixed arbitrary point that we are going to add to the route
                start_or_end: a string indicating if this new point is added to the start or end of the route.
        return: the calculated timestamp for the artifical start or end point
        '''
        #to get a more precise timestamp value of the new arbitrary fixed point that we are adding to the route based on the distance between this point and the first/last point as a
        #ratio of the total distance of the original route. 
        #Formula = (current distance between new point and first/last point) / total distance of original route * total duration of the route 
        tot_time = period[-1] - period[0]
        last_pt = (lats[-1], lons[-1])
        first_pt = (lats[0], lons[0])
        tot_dist = self.calculate_distance(first_pt, last_pt)
        if start_or_end == 'start':
            curr_dist = self.calculate_distance(new_pt, first_pt)
        elif start_or_end == 'end':
            curr_dist = self.calculate_distance(last_pt, new_pt)
        curr_time = curr_dist / tot_dist * tot_time
        if start_or_end == 'start':
            return period[0] - curr_time
        elif start_or_end == 'end':
            return period[-1] + curr_time

    def calculate_distance(self, coord1, coord2):
        '''
        @param: coord1: tuple of latitude and longitude for the first point. (latitude1, longitude1)
                coord2: tuple of latitude and longitude for the second point. (latitude2, longitude2)
        return: the numerical distance between 2 points
        '''
        #calculate the distance between 2 points
        return hs.haversine(coord1,coord2)

    def add_end_pt(self, dataframe):
        '''
        @param: dataframe: 1 row represents a trajectory of points

        return: a dataframe where a row represents a trajectory
        '''
        #check end pt
        lats = []
        lons = []
        for index, row in dataframe.iterrows():
            lats.append(row['Latitudes'][-1])
            lons.append(row['Longitudes'][-1])

        airport_name = dataframe['Dest'].unique()[0]
        bot_lat = self.airports_lat_lon[airport_name][0]
        bot_lon = self.airports_lat_lon[airport_name][1]

        idxs_extrapolated = self.check_end_lats_lons(dataframe, [], bot_lat, lats)
        idxs_extrapolated = self.check_end_lats_lons(dataframe, idxs_extrapolated, bot_lon, lons)

        #add end point to those indices
        update_idxes = dataframe.index.tolist()
        for j in range(len(idxs_extrapolated)):
            curr_idx = update_idxes[j]
            curr_ts = self.stage_1_csv.iloc[curr_idx, ]['Timestamps']
            curr_lat = self.stage_1_csv.iloc[curr_idx, ]['Latitudes']
            curr_lon = self.stage_1_csv.iloc[curr_idx, ]['Longitudes']
            curr_alt = self.stage_1_csv.iloc[curr_idx, ]['Altitudes']
            curr_head = self.stage_1_csv.iloc[curr_idx, ]['Headings']
            curr_spd = self.stage_1_csv.iloc[curr_idx, ]['Speeds']
            curr_num_pts = self.stage_1_csv.iloc[curr_idx, ]['Number of points']
            lat = deepcopy(curr_lat)
            lat.append(bot_lat)
            lon = deepcopy(curr_lon)
            lon.append(bot_lon)
            new_ts = self.calculate_time(curr_ts, curr_lat, curr_lon, (bot_lat, bot_lon), 'end')
            ts = deepcopy(curr_ts)
            ts.append(new_ts)
            alt = deepcopy(curr_alt)
            alt.append(curr_alt[-1]) #assume same values
            head = deepcopy(curr_head)
            head.append(curr_head[-1])
            spd = deepcopy(curr_spd)
            spd.append(curr_spd[-1])
            dataframe.at[curr_idx, 'Timestamps'] = ts
            dataframe.at[curr_idx, 'Latitudes'] = lat
            dataframe.at[curr_idx, 'Longitudes'] = lon
            dataframe.at[curr_idx, 'Altitudes'] = alt
            dataframe.at[curr_idx, 'Headings'] = head
            dataframe.at[curr_idx, 'Speeds'] = spd
            dataframe.at[curr_idx, 'Number of points'] += 1

        return dataframe

    def check_end_lats_lons(self, dataframe, idxs_extrapolated, thres, lats_lons):
        '''
        @param: dataframe: 1 row represents a trajectory of points.
                idxs_extrapolated: list of indices which have already been marked to add an arbitrary start point
                thres: threshold for latitude/longitude
                lats_lons: list of latitude/longitude values
        return: a list of indices where the trajectories have an artificial end point added
        '''
        #for those paths with ending lat/lon smaller than threshold, will add the threshold as the last point to the path. Only return indices
        update_idxes = dataframe.index.tolist()
        for j in range(len(lats_lons)):
            curr_lat = lats_lons[j]
            if curr_lat < thres - 0.01 or curr_lat > thres + 0.01: #needa add max as end pt
                curr_idx = update_idxes[j]
                if curr_idx not in idxs_extrapolated:
                    idxs_extrapolated.append(curr_idx)

        return idxs_extrapolated

    def interpolate_routes_in_segment(self, dataframe):
        '''
        @param: dataframe: 1 row represents a trajectory of points
        return: a dataframe where a row represents a trajectory
        '''
        #additional objective is to ensure the path connects to the artificial start/end point
        num_pts = dataframe['Number of points'].tolist()
        max_pts = self.num_ts

        for index, row in dataframe.iterrows(): #interpolate so that each route in the same segment will have the same number of points
            curr_num_pts = row['Number of points']
            curr_flight_id = row['Flight Id']
            curr_ts = row['Timestamps']
            period = np.linspace(curr_ts[0], curr_ts[-1], num = max_pts) 
            period = np.reshape(period, (1, max_pts)).tolist()[0]
            
            curr_lat = row['Latitudes']
            lat_vals = self.interpolate(period, curr_ts, curr_lat).tolist()

            curr_lon = row['Longitudes']
            lon_vals = self.interpolate(period, curr_ts, curr_lon).tolist()

            curr_alt = row['Altitudes']
            alt_vals = self.interpolate(period, curr_ts, curr_alt).tolist()

            curr_head = row['Headings']
            head_vals = self.interpolate(period, curr_ts, curr_head).tolist()

            curr_spd = row['Speeds']
            spd_vals = self.interpolate(period, curr_ts, curr_spd).tolist()

            dataframe.at[index, 'Timestamps'] = period
            dataframe.at[index, 'Latitudes'] = lat_vals
            dataframe.at[index, 'Longitudes'] = lon_vals
            dataframe.at[index, 'Altitudes'] = alt_vals
            dataframe.at[index, 'Headings'] = head_vals
            dataframe.at[index, 'Speeds'] = spd_vals
            dataframe.at[index, 'Number of points'] = max_pts

        return dataframe

    def interpolate(self, period, ts_lst, var_lst):
        '''
        @param: period: list of timestamps calculated from np.linspace function. This is used to calculate the corresponding attribute values at these timestamps. (0, 100, 200, ...)
                ts_lst: list of original timestamps in the route. This is for learning the function of the attributes.
                var_lst: list of original attribute values in the route. This represents the corresponding attribute values from the original timestamps. This is also for learning
                         the function of the attributes.
        return: a list of values representing the corresponding timestamps
        '''
        model = scipy.interpolate.interp1d(ts_lst, var_lst) #get the function

        values = model(period) #get the corresponding attributes for the given interpolated time period

        return values