import base64
import datetime
import io
import logging
import sys

import pandas as pd
from database import database_functions

import dash_core_components as dcc
import dash_html_components as html
import dash_table


import time
import sys

#GROUP_CONDITION IS HOW YOU IDENTIFY EACH TRAJECTORY
GROUP_CONDITION = 'temp_id'
class DashLogger(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream=stream)
        self.logs = list()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
            self.logs = self.logs[-1000:]
            self.flush()
        except Exception:
            self.handleError(record)

logger = logging.getLogger('display')
logger.setLevel(logging.INFO)
dash_logger = DashLogger(stream=sys.stdout)
logger.addHandler(dash_logger)

def parse_contents(filename):
    try:
        # filename = filename.split('.csv')[0]
        # _store_uploaded_file(filename,contents)
        _run_migration(filename.split('.csv')[0])
        # Assume that the user uploaded a CSV file
        
        # time.sleep(5)
        
    # elif 'xls' in filename:
    #     # Assume that the user uploaded an excel file
    #     df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        logger.warn('There was an error processing this file.')

def _run_migration(f_name):
    logger = logging.getLogger('display')
    logger.setLevel(logging.INFO)
    df = pd.read_csv('../csv/{}.csv'.format(f_name))
    df.datetime = pd.to_datetime(df.datetime)
    logger.info('==========================Preparing Data for Import...======')
    #preprocess here 
    _write_to_db(df)
    logger.info('===========================Data successfully imported!===============')


def _write_to_db(df):
    for index, group in df.groupby(GROUP_CONDITION):
        group = group.sort_values(by=['datetime'])
        if not group.datetime.is_unique:
            print ('{} is not unique in datetime..please check'.format(index))
            group.drop_duplicates(subset=['datetime'],inplace=True)
        component_length=len(group)
        datetime=group.iat[0,group.columns.get_loc('datetime')]
        external_id = str(index)
        track_id=database_functions.add_track_to_database(datetime,component_length,external_id)
        database_functions.add_trackpoints_to_database(group,track_id)