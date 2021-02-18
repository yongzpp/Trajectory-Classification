#!/usr/bin/python

import pyproj
import numpy as np
from dipy.segment.metric import Metric, ResampleFeature

#This class is for Clustering of paths in same segment (Stage 3)
class GPSDistance(Metric):
    def __init__(self):
        super().__init__(feature=ResampleFeature(nb_points = 1000)) #resample the given route to 1000 pts for better clustering results
        self._geod = pyproj.Geod(ellps='WGS84')
        self.distances = [] #to record the distances calculated

    def are_compatible(self, shape1, shape2):
        return len(shape1) == len(shape2)

    def dist(self, features_1, features_2):
        '''
        @param: features_1: longitudes and latitudes for the first route. [[longitude0, latitude0], [longitude1, latitude1], ...]
                features_2: longitudes and latitudes for the second route. [[longitude0, latitude0], [longitude1, latitude1], ...]
        '''
        # Euclidean GPS
        distances = self._geod.inv(features_1[:,0], features_1[:,1], features_2[:,0], features_2[:,1])[2]
        mean_dist = np.mean(distances)
        self.distances.append(mean_dist)
        return mean_dist #in meters