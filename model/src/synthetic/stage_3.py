#!/usr/bin/python
import pandas as pd
import numpy as np
from .GPSDistance import GPSDistance
from dipy.segment.clustering import QuickBundles

class stage_3:
    def __init__(self, stage_2_csv):
        #Flow:
        #1. Use quickbundles to cluster the paths first
        #2. Visualise the results and then make manual changes to the cluster labels of some of the paths
        self.stage_2_csv = stage_2_csv
        self.stage_3_csv = None


    #Stage 3: Clustering of paths in same segment
    def stage_3(self):
        '''
        return: a dataframe where a row represents a trajectory
        '''
        self.cluster_paths()
        return self.stage_3_csv

    def cluster_paths(self):
        '''
        final output: a dataframe where a row represents a trajectory
        '''
        self.stage_2_csv.reset_index(drop=True, inplace=True)
        dataframe = self.stage_2_csv
        unique_segments = dataframe['Segment Id'].unique()
        clusterids = [0] * dataframe.shape[0]
        cluster_counter = 0

        for i in unique_segments:
            sub_df = dataframe[dataframe['Segment Id'] == i]
            #cluster each segment
            routes = []
            for index, row in sub_df.iterrows():
                curr_ts = row['Timestamps']
                curr_lats = row['Latitudes']
                curr_lons = row['Longitudes']
                curr_alts = row['Altitudes']

                sub_routes = []
                for i in range(len(curr_ts)):
                    sub_routes.append([curr_lons[i], curr_lats[i]])
                sub_routes = np.array(sub_routes)
                sub_routes = sub_routes.reshape(len(curr_ts), 2)
                new_sub_routes = tuple(map(tuple, sub_routes))
                routes.append(sub_routes)

            routes = np.array(routes)
            metric, qb = self.initialise_qb(1) #get all distances first
            clusters = qb.cluster(routes)

            med = np.percentile(metric.distances, 25)

            metric, qb = self.initialise_qb(med)
            clusters = qb.cluster(routes)

            sub_df_idxs = sub_df.index.tolist()
            for i in clusters.get_large_clusters(1): #assign the rows to the different clusters
                idxes = i.indices
                for j in idxes:
                    curr_idx = sub_df_idxs[j]
                    clusterids[curr_idx] = cluster_counter
                cluster_counter += 1

        dataframe['Cluster Id'] = clusterids

        self.stage_3_csv = dataframe
        self.manually_alter_clusters()

    def initialise_qb(self, threshold):
        '''
        @param: threshold: an integer/float indicating the minimum distance for QuickBundles.
        return: metric: an object used to calculate the distance between the points
                qb: a Quickbundles object used to perform the clustering
        '''
        #create metric and quickbundles objects for clustering
        metric = GPSDistance()
        qb = QuickBundles(threshold = threshold, metric = metric) #a brute force iteration of all possible pairs of paths. Will first filter paths with distance above the threshold
        return metric, qb
    
    def manually_alter_clusters(self):
        '''
        final output: a dataframe where a row represents a trajectory
        '''
        #decided based on visualising the paths
        dataframe = self.stage_3_csv
        
        zeroes = [0, 1, 5, 6, 8, 9, 11, 14, 15, 16, 18, 19, 20, 21]
        ones = [2, 3, 4, 7, 10, 12, 13, 17]
        twos = [25, 29, 35, 38]
        threes = [22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 39, 40, 41, 42, 43]
        fours = [44, 46, 49, 50, 52, 54, 56, 57, 59, 62, 63, 65, 66, 67, 68, 69, 73, 74]
        fives = [45, 47, 48, 51, 53, 55, 58, 60, 61, 64, 70, 71, 72]
        sixes = [79, 107]
        sevens = [76, 77, 78, 80, 81, 83, 84, 85, 88, 89, 91, 92, 93, 94, 95, 98, 99, 100, 102, 103, 104, 109]
        eights = [75, 82, 86, 87, 90, 96, 97, 101, 105, 106, 108]

        lst = [zeroes, ones, twos, threes, fours, fives, sixes, sevens, eights]

        counter = 0
        for i in lst:
            for j in i:
                dataframe.at[j, 'Cluster Id'] = counter
            counter += 1

        dataframe = dataframe.drop([22, 23, 28, 33, 43]) #this is set manually
        dataframe.reset_index(drop=True, inplace=True)
        self.stage_3_csv = dataframe