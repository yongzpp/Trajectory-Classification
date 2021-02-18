#!/usr/bin/python
import pandas as pd
import numpy as np
from .create_config import cfg
from copy import deepcopy
import random

class stage_4:
    def __init__(self, stage_3_csv):
        #Flow:
        #1. Encode each path first based on the cluster id and dest
        #2. Retrieve the lat lon of the 3 airports
        #3. Retrieve and calculate the lat lon of the root coordinates which do not start from sg and prepare a dictionary to record the pair of lat lon (recall the 3 trees)
        #4. Calculate the change between 2 adjacent points for all paths
        #5. Combine different segments to generate all possible cases of synthetic trajectories up to length 2
        self.airports_lat_lon = cfg.preprocess.data.airports_lat_lon
        self.lat_lon_noise_mean = cfg.preprocess.data.lat_lon_noise_mean
        self.lat_lon_noise_sd = cfg.preprocess.data.lat_lon_noise_sd
        self.alt_noise_mean = cfg.preprocess.data.alt_noise_mean
        self.alt_noise_sd = cfg.preprocess.data.alt_noise_sd
        self.head_spd_noise_mean = cfg.preprocess.data.head_spd_noise_mean
        self.head_spd_noise_sd = cfg.preprocess.data.head_spd_noise_sd
        self.num_choices = cfg.preprocess.data.num_choices
        self.stage_3_csv = stage_3_csv
        self.stage_4_csv = None

    #Stage 4: sample joined trajectories
    def stage_4(self):
        '''
        return: a dataframe where a row represents a trajectory
        '''
        self.encode_new_labels()
        return self.stage_4_csv

    def encode_new_labels(self):
        '''
        final output: a dataframe where a row represents a trajectory
        '''
        #encode differently for the same segment (each cluster is a label), so it will be lat-lon-dest pattern-cluster id, where lat lon is the lat lon of the starting point of the current combination (recall the 3 tree diagrams and the translations!)
        #here, we will have dest pattern-cluster id
        dataframe = self.stage_3_csv
        unique_airports= dataframe['Dest'].unique()
        new_labels = [0] * dataframe.shape[0]

        for i in unique_airports:
            sub_df = dataframe[dataframe['Dest'] == i]

            unique_clusters = sub_df['Cluster Id'].unique()
            for j in range(len(unique_clusters)):
                clusterid = unique_clusters[j]
                sub_sub_df = sub_df[sub_df['Cluster Id'] == clusterid]
                sub_sub_df_idxes = sub_sub_df.index.tolist()
                curr_label = i + "-" + str(j)
                for k in sub_sub_df_idxes:
                    new_labels[k] = curr_label

        dataframe['Final Label'] = new_labels
        
        self.stage_4_csv = dataframe
        self.create_fixed_pts_of_categories()
    
    def create_fixed_pts_of_categories(self):
        '''
        final output: a dictionary where the key is a tuple, representing a pair of start point and end point, and the value is a list, representing the latitude and longitude values of the stated start point in the key 
        '''
        #To ensure that there will not be too many variations in the labels and to determine the root coordinates which do not start from sg (refer to the 3 tree diagrams)
        dataframe = self.stage_4_csv
        airports_lat_lon = self.airports_lat_lon
        unique_airports = list(airports_lat_lon.keys())
        unique_airports.remove('Singapore Changi Airport')
        sg_lat = airports_lat_lon['Singapore Changi Airport'][0]
        sg_lon = airports_lat_lon['Singapore Changi Airport'][1]

        mean = self.lat_lon_noise_mean
        sd = self.lat_lon_noise_sd * 5
        for i in unique_airports: #add noise so that the coordinates for the other airports are not translated to the other trees
            name = i+'2'
            noise = np.random.normal(mean, sd)
            i_lat = airports_lat_lon[i][0] + noise 
            noise = np.random.normal(mean, sd)
            i_lon = airports_lat_lon[i][1] + noise
            airports_lat_lon[name] = [i_lat, i_lon]

        unique_airports2 = [i+'2' for i in unique_airports]

        #create all possible combinations for all airports excluding sg
        for i in unique_airports2:
            i_lat = airports_lat_lon[i][0]
            i_lon = airports_lat_lon[i][1]
            for j in unique_airports:
                j_lat = airports_lat_lon[j][0]
                j_lon = airports_lat_lon[j][1]
                new_lat = i_lat + j_lat - sg_lat
                new_lon = i_lon + j_lon - sg_lon
                airports_lat_lon[(i, j)] = [new_lat, new_lon]

        self.airports_lat_lon = airports_lat_lon
        self.calculate_change_dataframe()

    def calculate_change_dataframe(self):
        '''
        final output: a dataframe where a row represents the change in adjacent attribute values in a trajectory
        '''
        #to calculate the change between 2 adjacent points for all attributes
        dataframe = self.stage_4_csv
        
        change_timestamps = []
        change_latitudes = []
        change_longitudes = []
        change_altitudes = []
        change_headings = []
        change_speeds = []
        change_labels = []
        change_segment_ids = []
        change_flight_ids = []
        change_dests = []

        for index, row in dataframe.iterrows():
            curr_timestamps = row['Timestamps']
            tmp = []
            for i in range(len(curr_timestamps)):
                if i == 0:
                    tmp.append(0) #reference point is the first point and we start from 0
                else:
                    tmp.append(curr_timestamps[i] - curr_timestamps[i - 1])
            change_timestamps.append(tmp)

            change_latitudes.append(self.calculate_difference(row['Latitudes']))
            change_longitudes.append(self.calculate_difference(row['Longitudes']))
            change_altitudes.append(self.calculate_difference(row['Altitudes']))
            change_headings.append(self.calculate_difference(row['Headings']))
            change_speeds.append(self.calculate_difference(row['Speeds']))

            curr_label = row['Final Label']
            change_labels.append(curr_label)
            change_segment_ids.append(row['Segment Id'])
            change_flight_ids.append(row['Flight Id'])
            change_dests.append(row['Dest'])

        change_df = pd.DataFrame() #useful for joining segments later
        change_df['Timestamps'] = change_timestamps
        change_df['Latitudes'] = change_latitudes
        change_df['Longitudes'] = change_longitudes
        change_df['Altitudes'] = change_altitudes
        change_df['Headings'] = change_headings
        change_df['Speeds'] = change_speeds
        change_df['Segment Id'] = change_segment_ids
        change_df['Flight Id'] = change_flight_ids
        change_df['Final Label'] = change_labels
        change_df['Dest'] = change_dests

        self.generate_trajectories(change_df)
    
    def calculate_difference(self, column): 
        '''
        @param: column: list of attribute values, like a list of timestamps, longitudes, etc.
        return: a list of values representing the change in attributes between adjacent points in a trajectory
        '''
        #will need to calculate the difference between 2 adjacent points. This is for combining the segments later.
        tmp = []
        for i in range(len(column)):
            if i == 0:
                tmp.append(column[i] - 0) #for the first point in the first segment. It acts as the reference point
            else:
                tmp.append(column[i] - column[i - 1]) 
        return tmp

    def generate_trajectories(self, change_df):
        '''
        @param: change_df: a dataframe where 1 row represents 1 trajectory of points, but each row will indicate the change in attribute values, not the original values
        final output: a dataframe where a row represents a combined trajectory
        '''
        dataframe = self.stage_4_csv
        
        #generate all cases
        unique_airports = ['Singapore Changi Airport']
        unique_airports.extend(dataframe['Dest'].unique().tolist())

        overall_timestamps = []
        overall_latitudes = []
        overall_longitudes = []
        overall_altitudes = []
        overall_headings = []
        overall_speeds = []
        overall_categories = []
        overall_segments = []
        overall_patterns = []
        overall_flightids = []

        for i in unique_airports:
            sub = deepcopy(unique_airports)
            if 'Singapore Changi Airport' in sub:
                sub.remove('Singapore Changi Airport')
            if i != 'Singapore Changi Airport': #this means that the root node is not sg
                i = i+'2'

            #length 1
            for j in sub:
                sub_df = dataframe[dataframe['Dest'] == j]
                unique_clusters = sub_df['Cluster Id'].unique()
                for k in unique_clusters:
                    clusters_df = sub_df[sub_df['Cluster Id'] == k]
                    row_idxs = clusters_df.index.tolist()
                    for s in range(self.num_choices):
                        chosen_row = random.choice(row_idxs) #1 trajectory

                        selected_rows = [chosen_row]
                        combined_timestamps = self.combining_segments(change_df, selected_rows, 'Timestamps', i)
                        combined_latitudes = self.combining_segments(change_df, selected_rows, 'Latitudes', i)
                        combined_longitudes = self.combining_segments(change_df, selected_rows, 'Longitudes', i)
                        combined_altitudes = self.combining_segments(change_df, selected_rows, 'Altitudes', i)
                        combined_headings = self.combining_segments(change_df, selected_rows, 'Headings', i)
                        combined_speeds = self.combining_segments(change_df, selected_rows, 'Speeds', i)
                        curr_pattern = dataframe.iloc[chosen_row, ]['Final Label']

                        lat_lon = self.airports_lat_lon[i]
                        curr_pattern = str(lat_lon[0]) + "-" + str(lat_lon[1]) + "-" + curr_pattern
                        combined_categories = [curr_pattern]
                        length_of_row = len(dataframe.iloc[chosen_row, ]['Timestamps'])
                        combined_patterns = [curr_pattern] * length_of_row

                        overall_timestamps.append(combined_timestamps)
                        overall_latitudes.append(combined_latitudes)
                        overall_longitudes.append(combined_longitudes)
                        overall_altitudes.append(combined_altitudes)
                        overall_headings.append(combined_headings)
                        overall_speeds.append(combined_speeds)
                        overall_categories.append(combined_categories)
                        overall_patterns.append(combined_patterns)
                        overall_segments.append(selected_rows)
                        overall_flightids.append([dataframe.iloc[chosen_row, ]['Flight Id']])

            #length 2
            chosen_rows = []
            for j in sub:
                sub_df = dataframe[dataframe['Dest'] == j]

                unique_clusters = sub_df['Cluster Id'].unique()
                for k in unique_clusters:
                    clusters_df = sub_df[sub_df['Cluster Id'] == k]
                    row_idxs = clusters_df.index.tolist()
                    for s in range(self.num_choices):
                        first_chosen_row = random.choice(row_idxs)
                        chosen_rows.append(first_chosen_row)

                        for b in sub:
                            new_sub_df = dataframe[dataframe['Dest'] == b]

                            unique_clusters = new_sub_df['Cluster Id'].unique()
                            for m in unique_clusters:
                                clusters_df = new_sub_df[new_sub_df['Cluster Id'] == m]
                                row_idxes = clusters_df.index.tolist()
                                second_chosen_row = random.choice(row_idxes)
                                #now, I will have a path of length 2
                                selected_rows = [first_chosen_row, second_chosen_row]
                                flightids = []
                                for n in selected_rows:
                                    flightids.append(dataframe.iloc[n, ]['Flight Id'])

                                combined_timestamps = self.combining_segments(change_df, selected_rows, 'Timestamps', i)
                                combined_latitudes = self.combining_segments(change_df, selected_rows, 'Latitudes', i)
                                combined_longitudes = self.combining_segments(change_df, selected_rows, 'Longitudes', i)
                                combined_altitudes = self.combining_segments(change_df, selected_rows, 'Altitudes', i)
                                combined_headings = self.combining_segments(change_df, selected_rows, 'Headings', i)
                                combined_speeds = self.combining_segments(change_df, selected_rows, 'Speeds', i)
                                combined_categories = []
                                combined_patterns = []

                                #first segment
                                curr_pattern = dataframe.iloc[first_chosen_row, ]['Final Label']
                                lat_lon = self.airports_lat_lon[i]
                                curr_pattern = str(lat_lon[0]) + "-" + str(lat_lon[1]) + "-" + curr_pattern
                                combined_categories.append(curr_pattern)
                                combined_patterns.extend([curr_pattern] * len(dataframe.iloc[first_chosen_row, ]['Timestamps']))

                                #second segment
                                curr_pattern = dataframe.iloc[second_chosen_row, ]['Final Label']
                                first_dest = dataframe.iloc[first_chosen_row, ]['Dest']
                                if i == 'Singapore Changi Airport':
                                    lat_lon = self.airports_lat_lon[first_dest]
                                else: #root node is not sg
                                    lat_lon = self.airports_lat_lon[(i, first_dest)]
                                curr_pattern = str(lat_lon[0]) + "-" + str(lat_lon[1]) + "-" + curr_pattern
                                combined_categories.append(curr_pattern)
                                combined_patterns.extend([curr_pattern] * (len(dataframe.iloc[second_chosen_row, ]['Timestamps']) - 1))

                                overall_timestamps.append(combined_timestamps)
                                overall_latitudes.append(combined_latitudes)
                                overall_longitudes.append(combined_longitudes)
                                overall_altitudes.append(combined_altitudes)
                                overall_headings.append(combined_headings)
                                overall_speeds.append(combined_speeds)
                                overall_categories.append(combined_categories)
                                overall_patterns.append(combined_patterns)
                                overall_segments.append(selected_rows)
                                overall_flightids.append(flightids)

        combined_df = pd.DataFrame()
        combined_df['Timestamps'] = overall_timestamps
        combined_df['Latitudes'] = overall_latitudes
        combined_df['Longitudes'] = overall_longitudes
        combined_df['Altitudes'] = overall_altitudes
        combined_df['Headings'] = overall_headings
        combined_df['Speeds'] = overall_speeds
        combined_df['Categories'] = overall_categories
        combined_df['Segments'] = overall_segments
        combined_df['Flight Ids'] = overall_flightids
        combined_df['Patterns'] = overall_patterns

        self.stage_4_csv = combined_df
    
    def combining_segments(self, dataframe, selected_segments, attribute, start):
        '''
        @param: dataframe: 1 row represents a trajectory of points
                selected_segments: a list of numbers which represents the indices of the selected rows in the dataframe. These rows will then be combined to form a trajectory
                attribute: a string indicating the column of interest (for example, longitude, latitude, etc). The segments will be combined 1 attribute value at a time.
                start: a string which represents the first airport. Note that it is not always changi airport.
        return: a list of values representing the attribute values after combining
        '''
        start_lat = self.airports_lat_lon[start][0]
        start_lon = self.airports_lat_lon[start][1]
        tmp = []
        for j in range(len(selected_segments)):
            change_segment = dataframe.iloc[selected_segments[j], ][attribute] #retrieve the segment from the dataframe
            if not tmp:
                if attribute == 'Latitudes' and start != 'Singapore Changi Airport': #start from the selected pt just now
                    last_time = start_lat
                    tmp = [start_lat]
                elif attribute == 'Longitudes' and start != 'Singapore Changi Airport':
                    last_time = start_lon
                    tmp = [start_lon]
                else:
                    last_time = 0 #if empty, this refers to the first segment
            else:
                last_time = tmp[-1]
            if j == 0 and start != 'Singapore Changi Airport' and not tmp: #first point of chosen segment is not relevant since the starting pt is not at sg. This is for non lat/lon attributes
                tmp = [change_segment[0]]
                last_time = tmp[-1]
            if j > 0 or start != 'Singapore Changi Airport': #for paths not starting with sg, first point is not relevant since it has already been added above
                change_segment = change_segment[1:] #trim off the first point of the new segment if it is a second segment and so on since we are going to merge segments
            for k in change_segment:
                tmp.append(last_time + k) #since it is a change
                last_time = tmp[-1]
        return tmp