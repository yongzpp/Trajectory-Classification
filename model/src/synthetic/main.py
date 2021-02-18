#!/usr/bin/python
from .stage_1 import stage_1
from .stage_2 import stage_2
from .stage_3 import stage_3
from .stage_4 import stage_4
from .stage_5 import stage_5
from .stage_6 import stage_6
from .create_config import cfg
from database import database_functions as db_func

import time

#Stage 1: remove segment routes
#Stage 2: interpolate trajectories same segment same length
#Stage 3: clustering of paths in same segment
#Stage 4: sample joined trajectories
#Stage 5: sample interpolated trajectories
#Stage 6: convert trajectories to points for database where a row represents a point
def run_main():

    print("Creating the synthetic dataset now for the first time...")
    tot_stages = cfg.preprocess.data.tot_stages

    t0 = time.time()
    counter = 1
    print("Stage " + str(counter) + " out of " + str(tot_stages) + ": " + "Reading the original file and filtering destinations...")
    counter += 1
    s1 = stage_1()
    stage_1_csv = s1.stage_1()
    t1 = time.time()
    print("Stage 1 done. Time taken: " + str(int(t1 - t0)) + "s")

    t2 = time.time()
    print("Stage " + str(counter) + " out of " + str(tot_stages) + ": " + "Interpolating trajectories in the same segment to have the same length...")
    counter += 1
    s2 = stage_2(stage_1_csv)
    stage_2_csv = s2.stage_2()
    t3 = time.time()
    print("Stage 2 done. Time taken: " + str(int(t3 - t2)) + "s")
    
    t4 = time.time()
    print("Stage " + str(counter) + " out of " + str(tot_stages) + ": " + "Clustering paths in each segment...")
    counter += 1
    s3 = stage_3(stage_2_csv)
    stage_3_csv = s3.stage_3()
    t5 = time.time()
    print("Stage 3 done. Time taken: " + str(int(t5 - t4)) + "s")
    
    t6 = time.time()
    print("Stage " + str(counter) + " out of " + str(tot_stages) + ": " + "Combining routes from different segments...")
    counter += 1
    s4 = stage_4(stage_3_csv)
    stage_4_csv = s4.stage_4()
    t7 = time.time()
    print("Stage 4 done. Time taken: " + str(int(t7 - t6)) + "s")
    
    t8 = time.time()
    print("Stage " + str(counter) + " out of " + str(tot_stages) + ": " + "Interpolating each combined trajectory to the same length...")
    counter += 1
    s5 = stage_5(stage_4_csv)
    stage_5_csv = s5.stage_5()
    t9 = time.time()
    print("Stage 5 done. Time taken: " + str(int(t9 - t8)) + "s")
    
    t10 = time.time()
    print("Stage " + str(counter) + " out of " + str(tot_stages) + ": " + "Converting interpolated but not normalised joined trajectories dataset to points for database...")
    counter += 1
    s6 = stage_6(stage_5_csv)
    stage_6_csv = s6.stage_6()
    #stage_6_csv.to_csv('Points_to_Database.csv', index = False)
    db_func.write(stage_6_csv, 'TRACKPOINTS')
    t11 = time.time()
    print('Wrote Points_to_Database to database container. Stage 6 done. Time taken: ' + str(int(t11 - t10)) + "s")
    