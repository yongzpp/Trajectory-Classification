#!/usr/bin/python
import pandas as pd
import numpy as np
from .create_config import cfg

class stage_1:
    def __init__(self):
        #Flow:
        #1. Filter for dests which have the desired number of paths
        #2. Remove outlier paths in each segment with start/end point out of 3 * sd range
        self.original_file = cfg.preprocess.data.file_name
        self.flight_counts = cfg.preprocess.data.flight_counts
        self.stage_1_csv = None


    #Stage 1: Removing segment routes
    def stage_1(self):
        '''
        return: a dataframe where a row represents a trajectory
        '''
        self.read_original_csv()
        return self.stage_1_csv

    def read_original_csv(self):
        self.stage_1_csv = pd.read_csv(self.original_file)
        self.get_dests()

    def get_dests(self):
        '''
        final output: a list of tuples, where a tuple represents an origin and dest, for example, [(origin1, dest1), (origin2, dest2), ...]
        '''
        #select dests with the desired number of paths
        dataframe = self.stage_1_csv
        unique_dests = dataframe['Destination'].unique()
        dests_num_routes = {}

        for i in unique_dests: #count the number of unique flights for each destination
            sub_df = dataframe[dataframe['Destination'] == i] 
            dests_num_routes[i] = len(sub_df['FlightId'].unique())

        dests_num_routes = list(dests_num_routes.items())
        dests_num_routes = sorted(dests_num_routes, key = lambda x: x[1]) #sort the destinations by number of unique
        #flights from smallest to biggest

        counts = list(map(lambda x: x[1], dests_num_routes))

        dests = list(filter(lambda x: x[1] in self.flight_counts, dests_num_routes)) #only want destinations with the desired number of flights

        final_origins_dest_pairs = []
        for i in dests: #there are 4 dests
            final_origins_dest_pairs.append(('Singapore Changi Airport', i[0]))

        self.filter_dests(final_origins_dest_pairs)

    def filter_dests(self, dest_pairs):
        '''
        @param: dest_pairs: a list of tuples of origin and dest of interest. [('Singapore Changi Airport', dest1), ('Singapore Changi Airport', dest2), ...]

        final output: a dataframe where a row represents a trajectory
        ''' 
        dataframe = self.stage_1_csv
        final_origins_dest_pairs = dest_pairs
        
        overall_timestamps = []
        overall_latitudes = []
        overall_longitudes = []
        overall_altitudes = []
        overall_headings = []
        overall_speeds = []
        overall_classids = []
        overall_flightids = []
        overall_starts = []
        overall_dests = []
        overall_num_pts = []

        classids = 0 #assign a unique segment id to each origin and dest pair

        for pair in final_origins_dest_pairs:
            sub_df = dataframe[(dataframe['Origin'] == pair[0]) & (dataframe['Destination'] == pair[1])]
            flight_ids = sub_df.FlightId.unique() #there are multiple paths from the same origin to same dest pair 

            for i in flight_ids:
                sub_flight_df = sub_df.loc[(sub_df['FlightId'] == i)] #subset each flight id from the dataframe
                timestamps = []
                latitudes = []
                longitudes = []
                altitudes = []
                headings = []
                speeds = []
                for index, row in sub_flight_df.iterrows(): #objective is to have 1 row as 1 route, each column will then be a list of the attributes for each point in the route
                    timestamps.append(row['Timestamp'])
                    latitudes.append(row['Latitude'])
                    longitudes.append(row['Longitude'])
                    altitudes.append(row['Altitude'])
                    headings.append(row['Heading'])
                    speeds.append(row['Speed'])

                overall_timestamps.append(timestamps)
                overall_latitudes.append(latitudes)
                overall_longitudes.append(longitudes)
                overall_altitudes.append(altitudes)
                overall_headings.append(headings)
                overall_speeds.append(speeds)
                overall_flightids.append(i)
                overall_classids.append(classids) #different routes in the same segment will still have the same segment id
                overall_starts.append(pair[0])
                overall_dests.append(pair[1])
                overall_num_pts.append(len(timestamps))

            classids += 1

        overall_df = pd.DataFrame()
        overall_df['Timestamps'] = overall_timestamps
        overall_df['Latitudes'] = overall_latitudes
        overall_df['Longitudes'] = overall_longitudes
        overall_df['Altitudes'] = overall_altitudes
        overall_df['Headings'] = overall_headings
        overall_df['Speeds'] = overall_speeds
        overall_df['Segment Id'] = overall_classids
        overall_df['Flight Id'] = overall_flightids
        overall_df['Origin'] = overall_starts
        overall_df['Dest'] = overall_dests
        overall_df['Number of points'] = overall_num_pts

        self.stage_1_csv = overall_df
        self.initial_remove_outliers()

    def initial_remove_outliers(self):
        '''
        final output: a dataframe, where a row represents a trajectory, with no extreme outlier paths for each segment
        '''
        # Remove the outliers in each segment first
        dataframe = self.stage_1_csv
        unique_segment_ids = dataframe['Segment Id'].unique()
        idxs_removed = []

        for i in unique_segment_ids:
            sub_df = dataframe[dataframe['Segment Id'] == i]

            #check start point
            idxs_removed.extend(self.check_indices(sub_df, 0))

            #check end point
            idxs_removed.extend(self.check_indices(sub_df, -1))

        update_overall_df = dataframe.drop(idxs_removed)
        self.stage_1_csv = update_overall_df


    def check_indices(self, dataframe, start_or_end):
        '''
        @param: dataframe: a dataframe where each row represents a trajectory of points
                start_or_end: a number to indicate if we are checking the first point or the last point of the route. If it is the first point, then this value will be 0. 
                              If it is the last point, then this value will be -1.

        return: returns a list of indices (which represent paths) to be removed as they are outliers
        '''               
        #Records the row indexes of paths with starting or ending point outside of 3 * sd range of the corresponding 25th/75th percentile
        idxs = []

        lats = []
        lons = []
        for index, row in dataframe.iterrows():
            curr_lat = row['Latitudes']
            curr_lon = row['Longitudes']
            lats.append(curr_lat[start_or_end])
            lons.append(curr_lon[start_or_end])

        bot_lat = np.percentile(lats, 25)
        med_lat = np.percentile(lats, 50)
        top_lat = np.percentile(lats, 75)
        sd_lat = np.std(lats)

        bot_lon = np.percentile(lons, 25)
        med_lon = np.percentile(lons, 50)
        top_lon = np.percentile(lons, 75)
        sd_lon = np.std(lons)

        for index, row in dataframe.iterrows():
            curr_lat = row['Latitudes']
            curr_lon = row['Longitudes']
            if curr_lat[start_or_end] < bot_lat - 3*sd_lat or curr_lat[start_or_end] > top_lat + 3*sd_lat: #remove if lat or lon out of 3*sd
                idxs.append(index)
            elif curr_lon[start_or_end] < bot_lon - 3*sd_lon or curr_lon[start_or_end] > top_lon + 3*sd_lon:
                idxs.append(index)

        return idxs
    