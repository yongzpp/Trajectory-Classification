#!/usr/bin/python
import pandas as pd
import numpy as np
from .create_config import cfg
import scipy.interpolate
import random

class stage_5:
    def __init__(self, stage_4_csv):
        #Flow:
        #1. Randomly determine to remove start or end or both of the trajectory
        #2. Randomly determine to remove middle of trajectory
        #3. Ensure distance between adjacent points are randomly uneven in a trajectory
        #4. Interpolate within each trajectory
        #5. Add noise to all atribute values to each trajectory
        self.lat_lon_noise_mean = cfg.preprocess.data.lat_lon_noise_mean
        self.lat_lon_noise_sd = cfg.preprocess.data.lat_lon_noise_sd
        self.alt_noise_mean = cfg.preprocess.data.alt_noise_mean
        self.alt_noise_sd = cfg.preprocess.data.alt_noise_sd
        self.head_spd_noise_mean = cfg.preprocess.data.head_spd_noise_mean
        self.head_spd_noise_sd = cfg.preprocess.data.head_spd_noise_sd
        self.start_end_min_frac_to_drop = cfg.preprocess.data.start_end_min_frac_to_drop
        self.start_end_max_frac_to_drop = cfg.preprocess.data.start_end_max_frac_to_drop
        self.frac_to_remove = cfg.preprocess.data.frac_to_remove
        self.num_ts = cfg.preprocess.data.num_ts
        self.min_num = cfg.preprocess.data.min_num
        self.max_num = cfg.preprocess.data.max_num
        self.stage_4_csv = stage_4_csv
        self.stage_5_csv = None

    #Stage 5: sample interpolated trajectories
    def stage_5(self):
        '''
        return: a dataframe where a row represents a trajectory
        '''
        self.interpolate_joined_trajectories()
        return self.stage_5_csv

    def interpolate_joined_trajectories(self):
        '''
        final output: a dataframe where a row represents a trajectory
        '''
        #Ensure each combined trajectory has varying number of points (to feed into model). The most tricky part is to decide on the final pattern.
        dataframe = self.stage_4_csv
        overall_ts = []
        overall_lat = []
        overall_lon = []
        overall_alt = []
        overall_head = []
        overall_spd = []
        overall_seg = []
        overall_flight_ids = []
        overall_pat = []

        for index, row in dataframe.iterrows():
            curr_ts = row['Timestamps']
            curr_lat = row['Latitudes']
            curr_lon = row['Longitudes']
            curr_alt = row['Altitudes']
            curr_head = row['Headings']
            curr_spd = row['Speeds']
            curr_seg = row['Segments']
            curr_flight_ids = row['Flight Ids']
            curr_cat = row['Categories']
            curr_pat = row['Patterns']
            ts_count = curr_ts[0] #timestamp counter
            ts_lst = [] #record the maximum timestamp for each segment. Since the segments are listed in order, we can increment the counter (seg_count) after we recorded the maximum timestamp for the previous segment.           
            
            idxs_to_remove = self.chop_start_end(curr_ts)
            #remove the indices to be chopped for start/end
            curr_ts = [curr_ts[i] for i in range(len(curr_ts)) if i not in idxs_to_remove]
            curr_lat = [curr_lat[i] for i in range(len(curr_lat)) if i not in idxs_to_remove]
            curr_lon = [curr_lon[i] for i in range(len(curr_lon)) if i not in idxs_to_remove]
            curr_alt = [curr_alt[i] for i in range(len(curr_alt)) if i not in idxs_to_remove]
            curr_head = [curr_head[i] for i in range(len(curr_head)) if i not in idxs_to_remove]
            curr_spd = [curr_spd[i] for i in range(len(curr_spd)) if i not in idxs_to_remove]
            curr_pat = [curr_pat[i] for i in range(len(curr_pat)) if i not in idxs_to_remove]
            
            res = self.chop_middle(curr_ts) #randomly determine if need to remove a block of the timestamps
            if res: 
                curr_ts = [curr_ts[i] for i in range(len(curr_ts)) if i not in res] #filter this block
                curr_lat = [curr_lat[i] for i in range(len(curr_lat)) if i not in res]
                curr_lon = [curr_lon[i] for i in range(len(curr_lon)) if i not in res]
                curr_alt = [curr_alt[i] for i in range(len(curr_alt)) if i not in res]
                curr_head = [curr_head[i] for i in range(len(curr_head)) if i not in res]
                curr_spd = [curr_spd[i] for i in range(len(curr_spd)) if i not in res]
                curr_pat = [curr_pat[i] for i in range(len(curr_pat)) if i not in res]
            
            seg_flight_ids = {curr_cat[0]:[curr_seg[0], curr_flight_ids[0]]} #record segment and flight id for each pattern
            counter = 1
            for i in curr_cat[1:]: #record from first segment
                curr_idx = curr_pat.index(i)
                las_ts = curr_ts[curr_idx - 1]
                ts_count += las_ts
                ts_lst.append(las_ts)
                
                pat_seg = curr_seg[counter]
                pat_flightid = curr_flight_ids[counter]
                seg_flight_ids[i] = [pat_seg, pat_flightid]
                counter += 1
            ts_lst.append(curr_ts[-1])
            period = np.linspace(curr_ts[0], curr_ts[-1], num = self.num_ts) #interpolate to get the desired number of timestamps for each route
            period = np.reshape(period, (1, self.num_ts)).tolist()[0]
            
            for i in range(1, len(period) - 1):
               curr = period[i]
               prev = period[i-1]
               nxt = period[i+1]
               new = random.uniform(prev + 1, nxt - 1) #add noise so that the interval is not fixed
               period[i] = new
            
            pats_num = [] #record the final pattern for the route
            for i in period: #linear scan through the period. So if the current element (timestamp) is smaller than or equal to the maximum timestamp for this segment, then we know that it must belong to this segment. Else, we will move on and check the next segment
                curr_idx = 0
                while curr_idx < len(ts_lst):
                    current_ts = ts_lst[curr_idx]
                    if current_ts >= i:
                        break
                    else:
                        curr_idx += 1
                curr_idx = min(curr_idx, len(ts_lst) - 1)
                pats_num.append(curr_cat[curr_idx])

            interpolated_lat_values = self.interpolate(period, curr_ts, curr_lat).tolist() #get the interpolated attributes
            interpolated_lat_values = self.lat_lon_noise(interpolated_lat_values, pats_num)
            interpolated_lon_values = self.interpolate(period, curr_ts, curr_lon).tolist()
            interpolated_lon_values = self.lat_lon_noise(interpolated_lon_values, pats_num)
            interpolated_alt_values = self.interpolate(period, curr_ts, curr_alt).tolist()
            interpolated_alt_values = self.alt_noise(interpolated_alt_values)
            interpolated_head_values = self.interpolate(period, curr_ts, curr_head).tolist()
            interpolated_head_values = self.head_spd_noise(interpolated_head_values)
            interpolated_spd_values = self.interpolate(period, curr_ts, curr_spd).tolist()
            interpolated_spd_values = self.head_spd_noise(interpolated_spd_values)
            
            #randomly sample the number of points for this trajectory
            random_num = random.randint(self.min_num, self.max_num)
            idxs = np.random.choice(list(range(self.num_ts)), size = random_num, replace = False)
            idxs.sort()
            #filter the idxes from the original lists
            period = [period[i] for i in range(len(period)) if i not in idxs]
            interpolated_lat_values = [interpolated_lat_values[i] for i in range(len(interpolated_lat_values)) if i not in idxs]
            interpolated_lon_values = [interpolated_lon_values[i] for i in range(len(interpolated_lon_values)) if i not in idxs]
            interpolated_alt_values = [interpolated_alt_values[i] for i in range(len(interpolated_alt_values)) if i not in idxs]
            interpolated_head_values = [interpolated_head_values[i] for i in range(len(interpolated_head_values)) if i not in idxs]
            interpolated_spd_values = [interpolated_spd_values[i] for i in range(len(interpolated_spd_values)) if i not in idxs]
            pats_num = [pats_num[i] for i in range(len(pats_num)) if i not in idxs]
            
            tmp_seg = []
            tmp_flight_ids = []
            unique_pats = list(set(pats_num))
            for i in unique_pats: #some patterns could be lost due to the random sampling of points
                tmp_seg.append(seg_flight_ids[i][0])
                tmp_flight_ids.append(seg_flight_ids[i][1])
            curr_seg = tmp_seg
            curr_flight_ids = tmp_flight_ids


            overall_ts.append(period)
            overall_lat.append(interpolated_lat_values)
            overall_lon.append(interpolated_lon_values)
            overall_alt.append(interpolated_alt_values)
            overall_head.append(interpolated_head_values)
            overall_spd.append(interpolated_spd_values)
            overall_seg.append(curr_seg)
            overall_flight_ids.append(curr_flight_ids)          
            overall_pat.append(pats_num)

        overall_df = pd.DataFrame()
        overall_df['Timestamps'] = overall_ts
        overall_df['Latitudes'] = overall_lat
        overall_df['Longitudes'] = overall_lon
        overall_df['Altitudes'] = overall_alt
        overall_df['Headings'] = overall_head
        overall_df['Speeds'] = overall_spd
        overall_df['Segments'] = overall_seg
        overall_df['Flight Ids'] = overall_flight_ids
        overall_df['Patterns'] = overall_pat
        
        self.stage_5_csv = overall_df

    def chop_start_end(self, period):
        '''
        @param: period: list of timestamp values
        return: a list of indices where points are removed either at the start or at the end or both of the trajectory
        '''
        #returns idxs to be removed from trajectory
        idxs = []
        length = len(period)
        start_ratio = round(random.uniform(0, 1), 1)
        if start_ratio >= 0.5:
            min_num_pts_to_drop = int(self.start_end_min_frac_to_drop * length)
            max_num_pts_to_drop = int(self.start_end_max_frac_to_drop * length)
            num_pts_to_drop = random.randint(min_num_pts_to_drop, max_num_pts_to_drop)
            idxs.extend(list(range(num_pts_to_drop)))
        end_ratio = round(random.uniform(0, 1), 1)
        if end_ratio >= 0.5:
            min_num_pts_to_drop = int(self.start_end_min_frac_to_drop * len(period))
            max_num_pts_to_drop = int(self.start_end_max_frac_to_drop * len(period))
            num_pts_to_drop = random.randint(min_num_pts_to_drop, max_num_pts_to_drop)
            lst = list(range(len(period) - num_pts_to_drop, length + 1))
            idxs.extend(lst)
        return idxs

    def chop_middle(self, period):
        '''
        @param: period: list of timestamp values
        return: If there are points removed in the middle of the trajectory, return the list of indices. Else, return False
        '''
        #returns idxs to be removed from trajectory
        chop_ratio = round(random.uniform(0, 1), 1)
        if chop_ratio >= 0.5:
            num_ts_to_drop = int(self.frac_to_remove * len(period)) #to remove a block of timestamps in the middle
            start = random.randint(num_ts_to_drop, len(period) - 2 * num_ts_to_drop)
            idxs_chopped = list(range(start, start + num_ts_to_drop))
            return idxs_chopped
        return False

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

    def lat_lon_noise(self, lat_lon, pat):
        '''
        @param: lat_lon: list of longitude or latitude values
                pat: list of corresponding pattern labels
        return: a list of longitude or latitude values with added noise every 3 points
        '''
        mean = self.lat_lon_noise_mean
        sd = self.lat_lon_noise_sd
        noise = np.random.normal(mean, sd, len(lat_lon))
        for i in range(len(lat_lon)):
            if i % 3 == 0 and 'Clark' not in pat[i]: #do not add noise to angeles clark airport
                lat_lon[i] += noise[i]
        return lat_lon

    def alt_noise(self, alt):
        '''
        @param: alt: list of altitude values
        return: a list of altitude values with added noise every 3 points
        '''
        mean = self.alt_noise_mean
        sd = self.alt_noise_sd
        noise = np.random.normal(mean, sd, len(alt))
        for i in range(len(alt)):
            if i % 3 == 0:
                alt[i] += noise[i]
        return alt

    def head_spd_noise(self, head_spd):
        '''
        @param: head_spd: list of heading or speed values
        return: a list of head/speed values with added noise every 3 points
        '''
        mean = self.head_spd_noise_mean
        sd = self.head_spd_noise_sd
        noise = np.random.normal(mean, sd, len(head_spd))
        for i in range(len(head_spd)):
            if i % 3 == 0:
                head_spd[i] += noise[i]
        return head_spd