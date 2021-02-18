#%%
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import plotly.figure_factory as ff
from .app import app,db
import plotly.graph_objs as go
from plotly import tools
import pandas as pd
from datetime import datetime as dt
from datetime import date, timedelta
from datetime import datetime
import numpy as np
import copy

from database import database_functions
from . import figures
from . import functions
from . import config as cfg
from . import uploader
from . import pages


ops_cols = ['asset',
        'fob',
        'flight_plan',
        'flight_response',
        'kri_cord',
        'vessel_report',
        'transponder_off',
        'diversion',
        'airspace_infringe',
        'saf_interfere',
        'hhq_call',
        'service_cord',
    ]

def is_extention(row):
    return ('new' not in row.callsign) or ('new' not in row.squadcode) 

def get_rank(row):
    if ('new_' not in row.callsign) and ('new_' not in row.squadcode):
        return 2
    elif ('new_' in row.callsign) and ('new_' in row.squadcode):
        return 0
    else:
        return 1


@app.callback(Output('main_tab_content', 'children'),
              [Input('main_tabs', 'value')])
def render_content(tab):
    if tab == 'tab_viz':
        return pages.get_viz_page()
    elif tab == 'tab_upload':
        return pages.get_upload_page()
    elif tab =='tab_summary':
        return pages.get_summary_page()
    elif tab == 'tab_real':
        return pages.get_real_page()


########################map stuff##########################3
@app.callback(Output('map', 'figure'),
[Input('tracksid', 'value'),Input('map_options', 'value'),],
	[State('map','figure')])
def update_map(track_ids,map_options,old_map): #for the tab showing only real data
    '''
    @param: track_ids: list of trackids selected from the dropdown
            map_options: show directional arrows or not
            old_map: old state of the map (don't care)
    '''
    df = database_functions.read_unique_tracks(track_ids)
    figure = figures.generate_map_figure(df,cfg.OFFLINE,map_options)
    return figure

@app.callback(Output('mappred', 'figure'),
[Input('tracksidpred', 'value'),Input('map_optionspred', 'value'),],
    [State('mappred','figure')])
def update_mappred(track_ids,map_options,old_map): #for the tab showing real data + predictions
    '''
    @param: track_ids: list of trackids selected from the dropdown
            map_options: show directional arrows or not
            old_map: old state of the map (don't care)
    '''
    df = pd.DataFrame()
    actual_df = pd.DataFrame()
    found_pred = False
    found_actual = False
    pred_track_ids = []
    actual_track_ids = []
    for i in track_ids: #to check if there is any actual/predicted flight selected, split into the corresponding lists
        if 'Actual' in i:
            found_actual = True
            actual_track_ids.append(i)
        elif 'Predicted' in i:
            found_pred = True
            pred_track_ids.append(i)
    #track_ids = [i.replace('(Actual)', '') for i in track_ids]
    #track_ids = [i.replace('(Predicted)', '') for i in track_ids]
    #track_ids = [int(i) for i in track_ids]
    #print(track_ids)
    #print(pred_track_ids)
    #print(actual_track_ids)
    if found_pred: #retrieve individually
        df = database_functions.read_unique_trackspred(pred_track_ids)
    if found_actual:
        actual_df = database_functions.read_unique_tracks(actual_track_ids)

    '''
    if not df.empty: #got predictions
        try:
            df.columns = ['PointId', 'timestamp', 'latitude', 'longitude', 'altitude', 'heading', 'speed', 'track_id', 'Patterns'] #rename columns
            df['timestamp'] = df['timestamp'].apply(lambda x: float(x)) #convert to float for interpolation later
        except Exception:
            df.columns = ['timestamp', 'latitude', 'longitude', 'altitude', 'heading', 'speed', 'track_id', 'Patterns'] #test data csv fed directly
    '''
    #print(list(df.columns))
    #print(list(actual_df.columns))

    overall_df = pd.DataFrame()
    if not df.empty and actual_df.empty: #only predictions, no actual data
        df['track_id'] = df['track_id'].apply(lambda x: str(x) + '(Predicted)') #append so that can see on the map
        overall_df = df
    elif not actual_df.empty and df.empty: #only actual data, no predictions
        actual_df['track_id'] = actual_df['track_id'].apply(lambda x: str(x) + '(Actual)')
        overall_df = actual_df
    else: #got both actual data and predictions
        df['track_id'] = df['track_id'].apply(lambda x: str(x) + '(Predicted)')
        actual_df['track_id'] = actual_df['track_id'].apply(lambda x: str(x) + '(Actual)')
        #ids = list(df.PointId.values)
        trackids = list(df.track_id.values)
        latitudes = list(df.latitude.values)
        longitudes = list(df.longitude.values)
        altitudes = list(df.altitude.values)
        speeds = list(df.speed.values)
        #bearings = list(df.heading.values)
        bearings = list(df.bearing.values)
        #datetimes = list(df.timestamp.values)
        datetimes = list(df.datetime.values)
        datetimes = [i.tolist() for i in datetimes]
        patterns = list(df.Patterns.values)

        #actual_ids = list(actual_df.id.values)
        actual_trackids = list(actual_df.track_id.values)
        actual_latitudes = list(actual_df.latitude.values)
        actual_longitudes = list(actual_df.longitude.values)
        actual_altitudes = list(actual_df.altitude.values)
        actual_speeds = list(actual_df.speed.values)
        actual_bearings = list(actual_df.bearing.values)
        actual_datetimes = list(actual_df.datetime.values)
        actual_datetimes = [i.tolist() for i in actual_datetimes]
        actual_patterns = list(actual_df.new_label.values)

        new_df = pd.DataFrame()
        #new_df['id'] = ids
        new_df['track_id'] = trackids
        new_df['latitude'] = latitudes
        new_df['longitude'] = longitudes
        new_df['altitude'] = altitudes
        new_df['speed'] = speeds
        new_df['bearing'] = bearings
        new_df['datetime'] = datetimes
        new_df['new_label'] = patterns

        actual_new_df = pd.DataFrame()
        #actual_new_df['id'] = actual_ids
        actual_new_df['track_id'] = actual_trackids
        actual_new_df['latitude'] = actual_latitudes
        actual_new_df['longitude'] = actual_longitudes
        actual_new_df['altitude'] = actual_altitudes
        actual_new_df['speed'] = actual_speeds
        actual_new_df['bearing'] = actual_bearings
        actual_new_df['datetime'] = actual_datetimes
        actual_new_df['new_label'] = actual_patterns

        overall_df = pd.concat([new_df, actual_new_df]) #concat to form an overall df

        #print('here')
        #print(overall_df.track_id.unique())
    figure = figures.generate_map_figure(overall_df,cfg.OFFLINE,map_options)
    return figure


# @app.callback(Output('track_info_table', 'figure'),
# [Input('tracks', 'value')],
# 	[State('track_info_table','figure')])
# def update_track_info(track_id,old_figure):
#     if track_id != "None":
#         df = database_functions.read_track(track_id)
#         # df = df.drop(columns=['id','label','origin','destination','is_simulated','datetime','has_full','has_cs','has_sq','has_double','n_external_track_id'])
#         df = df[['external_id','component_length']]
#         df = df.rename(columns ={'component_length':'track length'})
#         figure = figures.generate_table_figure(df)
#         return figure
#     else:
#         old_figure['data']= []
#         return old_figure



for cord in ['latitude','longitude','altitude']:
    @app.callback(Output(cord, 'figure'),
    [Input('tracksid', 'value'),Input('map', 'hoverData')],
    [State(cord,'figure')]	)
    def update_cord(track_id,click_data,old_state,cord=cord): #showing the 3 graphs at the side, for the tab showing only real data
        '''
        @param: track_id: list of track ids
                click_data: for showing the vertical blue line in the graphs
                old_state: dont care
                cord: lat or lon or alt, a string
        '''
        df = database_functions.read_unique_tracks(track_id)
        figure = figures.generate_cord_figure(df,click_data,old_state,cord)
        return figure

    @app.callback(Output(cord + 'pred', 'figure'),
    [Input('tracksidpred', 'value'),Input('mappred', 'hoverData')],
    [State(cord + 'pred','figure')]  )
    def update_cordpred(track_id,click_data,old_state,cord=cord): #showing the 3 graphs at the side, for the tab showing real data + predictions
        '''
        @param: track_id: list of track ids
                click_data: for showing the vertical blue line in the graphs
                old_state: dont care
                cord: lat or lon or alt, a string
        '''
        df = database_functions.read_unique_trackspred(track_id)
        figure = figures.generate_cord_figure(df,click_data,old_state,cord)
        return figure
    
############################################################## extra ops options

@app.callback(
    Output('ops_extra_options', 'style'),
    [Input('label', 'value')],
	)
def update_ops_extra_options(label):
    if label == 'ops':
        return {'padding-top':'5px','padding-bottom':'5px'}
    else:
        return {'padding-top':'5px','padding-bottom':'5px','display': 'none'}
######################## update label
entry_cols = ['label','sublabel','origin','destination'] +  ops_cols 
state_value_cols = entry_cols + ['tracks']
update_states = [State(col,'value') for col in state_value_cols] 
@app.callback(Output('update_memory', 'data'),
    [Input('update_label_button', 'n_clicks')],
	update_states
	)    
def update_label(nclicks,*args):
    input_names = [item.component_id for item in update_states]
    kwargs_dict = dict(zip(input_names, args))
    if nclicks:
        if (not kwargs_dict['origin']) or kwargs_dict['origin'] == '':
            kwargs_dict['origin'] = 'None'
        if (not kwargs_dict['destination']) or kwargs_dict['destination'] == '':
            kwargs_dict['destination'] = 'None'
        try:
            query = database_functions.query_track(kwargs_dict['tracks'])
            for col in entry_cols:
                setattr(query,col,kwargs_dict[col])
            db.session.commit()
            data = {'update_success':True}
        except:
            data = {'update_success':False}
    else:
        data = None

    
    return data



@app.callback(Output('modal', 'is_open'),
             [Input('update_memory', 'data'),Input("modal_close", "n_clicks")],
             [State("modal", "is_open"),State('ops_option_memory', 'data')])
def open_log(data,closeclick,is_open,track_data):

    if data or closeclick:
        return not is_open
    return is_open

@app.callback(Output('modal_body', 'children'),
             [Input('update_memory', 'data')],
             )
def update_log_text(data,):
    if data:
        if data['update_success']:
            return 'Success! Track information updated!'
        else:
            return 'Track update failed. Please ensure entered data is valid and try again'
    else:
        return ''

#######################filter callbacks########################################

@app.callback(Output('label_filter', 'options'),
    [Input('update_memory', 'data')],
	[State('label_filter', 'options')]
    )    
def update_label2(data, old_options):
    if data:
        if data['update_success']:
            labels = database_functions.read_labels()['label'].unique()
            options = [{'label': 'All Labels','value': 'all'}]+[{'label': t,'value': t} for t in labels]
    else:
        options = old_options
    return options

@app.callback(Output('callsign_filter', 'options'),
    [Input('date_filter', 'value')],
    )    
def update_cs_option(date):
    labels = database_functions.read_callsign(date)['callsign']
    options = [{'label': 'All Callsigns','value': 'all'}]+[{'label': t,'value': t} for t in labels if 'new' not in t]
    return options

@app.callback(Output('squadcode_filter', 'options'),
    [Input('date_filter', 'value')],
    )    
def update_sq_option(date):
    labels = database_functions.read_squadcode(date)['squadcode']
    options = [{'label': 'All Mode3','value': 'all'}]+[{'label': t,'value': t} for t in labels if 'new' not in t]
    return options
######################################################################## 
##############################track options##########################
@app.callback(
    Output('tracks', 'options'),
    [Input('date_filter', 'value')],
	)
def update_tracks(date_filter): #update the patterns list in the dropdown, for the tab showing only real data
    '''
    @param: date_filter: date selected in the date dropdown, don't care
    '''
    df = database_functions.read_by_labels()
    uniquelabels = df['new_label'].unique()
    if len(df):
        df = df.sort_values(by=['datetime'],ascending=[True])
        options = [{
             'label': 'Label {}'.format(i),
           
            'value': i} for i in uniquelabels]  + [{'label':'All Labels','value':'all'}]
        return options
    else:
        options = [{'label':'No Track found, please check filters','value':'None','disabled':True}]
    return options

@app.callback(
    Output('trackspred', 'options'),
    [Input('date_filterpred', 'value')],
    )
def update_trackspred(date_filter): #update the patterns list in the dropdown, for the tab showing real data + predictions
    '''
    @param: date_filter: date selected in the date dropdown, don't care
    '''
    df = database_functions.read_by_labels()
    df = database_functions.read_by_labelspred()
    uniquepatterns = df['Patterns'].unique()
    if len(df):
        options = [{
            'label': 'Pattern {}'.format(i),
           
        'value': i} for i in uniquepatterns]
        return options
    else:
        options = [{'label':'No Track found, please check filters','value':'None','disabled':True}]
    return options

@app.callback(
    Output('tracksid', 'options'),
    [Input('tracks', 'value')],
    )
def update_tracksids(trackid): #update the tracks id list in the dropdown, for the tab showing real data
    '''
    @param: trackid: can be list of pattern ids or just a pattern id string
    '''
    df = database_functions.read_unique_labels(trackid)
    uniquelabels = df['track_id'].unique()

    if len(df):
        options = [{'label':'All Tracks','value':'all'}] + [{
             'label': 'Track Id {}'.format(i),
           
            'value': i} for i in uniquelabels]
        return options
    else:
        options = [{'label':'No Track Id found, please check filters','value':'None','disabled':True}]
    return options

@app.callback(
    Output('tracksidpred', 'options'),
    [Input('trackspred', 'value')],
    )
def update_tracksidspred(trackid): #update the tracks id list in the dropdown, for the tab real data + predictions
    '''
    @param: trackid: can be list of pattern ids or just a pattern id string
    '''
    df = database_functions.read_unique_labelspred(trackid) #retrieve data separately
    actual_df = database_functions.read_unique_labels(trackid)
    uniquelabels = df['track_id'].unique()
    uniquelabels = [str(i) + '(Predicted)' for i in uniquelabels] #append predicted for predictions
    actualuniquelabels = actual_df['track_id'].unique()
    actualuniquelabels = [str(i) + '(Actual)' for i in actualuniquelabels] #append actual for real data
    
    #print(uniquelabels)
    #print(actualuniquelabels)

    if len(df):
        options = [{'label':'All Tracks','value':'all'}] + [{ #concatenate both options
             'label': 'Track Id {}'.format(i),
           
            'value': i} for i in uniquelabels] + [{
             'label': 'Track Id {}'.format(i),
           
            'value': i} for i in actualuniquelabels]
        return options
    else:
        options = [{'label':'No Track Id found, please check filters','value':'None','disabled':True}]
    return options

@app.callback(
    Output('tracks', 'value'),
    [Input('tracks', 'options')],
	)
def update_track_value(options):
    if options:
        return options[0]['value']

@app.callback(
    Output('trackspred', 'value'),
    [Input('trackspred', 'options')],
    )
def update_track_valuepred(options):
    if options:
        return options[0]['value']

#################################################### value update after options
@app.callback(
    Output('callsign_filter', 'value'),
    [Input('callsign_filter', 'options')],
	)
def update_cs_value(options):
    if options:
        return options[0]['value']


@app.callback(
    Output('squadcode_filter', 'value'),
    [Input('squadcode_filter', 'options')],
	)
def update_sq_value(options):
    if options:
        return options[0]['value']

@app.callback(Output('label_filter', 'value'),
    [Input('label_filter', 'options')],
	[State('label_filter', 'value')]
    )
def update_label_filter_value(options,old_value):
    labels = database_functions.read_labels()['label'].unique()
    if old_value in labels:
        return old_value
    else:
        return options[0]['value']

@app.callback(
    Output('sublabel', 'value'),
    [Input('sublabel', 'options')],
	)
def update_sublabel_value(options):
    if options:
        return options[0]['value']
###################################################################################
# @app.callback(Output('upload_output', 'children'),
#               [Input('upload-data', 'contents')],
#               [State('upload-data', 'filename'),
#                State('upload-data', 'last_modified')])
# def update_upload(list_of_contents, list_of_names, list_of_dates):


#     if list_of_contents is not None:
#         list_of_contents = [list_of_contents]
#         list_of_dates = [list_of_dates]
#         list_of_names = [list_of_names]
#         children = [
#             uploader.parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children

    
@app.callback(Output('hidden_div', 'children'),[Input('upload_button', 'n_clicks')],
	[State('filelist_dropdown','value')])    
def upload(nclicks,filename):
    if nclicks!=None and filename!=None:
        uploader.parse_contents(filename)

@app.callback(
    Output('log', 'children'),
    [Input('log-update', 'n_intervals')])
def update_logs(interval):
    return [html.Div(log) for log in uploader.dash_logger.logs]


@app.callback(Output('update_label_button', 'disabled'),
             [Input('tracks', 'value')])
def set_button_enabled_state(track_id):
    return track_id =='None'



@app.callback(Output('no_track_log', 'children'),
             [Input('tracks', 'value')],

             )
def update_no_track(track_id):
    if track_id == "None":
        return 'No track Selected. Please select a track'
    # else:
    #     return ''


    # else:
    #     return ''
# @app.callback(Output('label_filter', 'options'),[Input('update_label_button', 'n_clicks')],
# 	)    
# def update_hidden_value(nclicks):
#     if nclicks!=None:
#         labels=read_labels()['label'].unique()
#         return [{'label':'All Labels','value':'all'}]+[{'label':t,'value':t} for t in labels]

############################################################ update ui using db info##########################################
@app.callback(Output('ops_option_memory', 'data'),
             [Input('tracks', 'value')])
def store_track_info(track_id):
    if track_id != "None":
        df = database_functions.read_track(track_id).iloc[0].to_dict()
        return df
    else:
        return None

@app.callback(Output('label', 'value'),
    [Input('ops_option_memory', 'data')],
    [State('label','value'),State('label','options')])
def update_default_label(data,old_label,old_options):
    if data:
        label = data['label']
        if label=='None':
            return old_options[0]['value']
        else:
            return label
    else:
        return old_label
@app.callback(Output('origin', 'value'),
[Input('ops_option_memory', 'data')],
    [State('origin','value')])
def update_default_origin(data,old_origin):
    if data:
        origin = data['origin']
        return origin
    else:
        return old_origin
@app.callback(Output('destination', 'value'),
[Input('ops_option_memory', 'data')],
    [State('destination','value')] )
def update_default_destination(data,old_dest):
    if data:
        dest = data['destination']
        return dest
    else:
        return old_dest
flight_subtypes = {
    'ops': {
        'JTF ALKI I Patrol': 'alki',
        'KOOPSAU I Air Patrol': 'koopsau',
        'KOHANUDNAS Air Defence Patrol': 'kohanudnas'
        },
    'training':{
        '1': '1',
        '2': '2',
        },
    'misc':{
        'Transport': 'transport',
        'Air-to-Air Refuel': 'a2a',
        'Others': 'others'
        }
    }
reverse_keys={}
for lab in flight_subtypes.keys():
    reverse_keys[lab] = {v:k for k,v in flight_subtypes[lab].items()}
@app.callback(
    Output('sublabel', 'options'),
    [Input('label', 'value')],
	[State('ops_option_memory','data')])
def update_subtype_options(label,data):
    options_dic = copy.deepcopy(flight_subtypes[label])
    option_list = [{'label':k,'value':v}for k,v in options_dic.items()]
    if data:
        sublabel = data['sublabel']
        if sublabel != 'None' and data['label'] == label:
            sublabel_key = reverse_keys[label][sublabel]
            print(sublabel_key)
            del options_dic[sublabel_key]
            option_list = [{'label':sublabel_key,'value':sublabel}]+ [{'label':k,'value':v}for k,v in options_dic.items()]
    return option_list 

##store updatre for ops extra



for col in ops_cols:
    @app.callback(Output(col, 'value'),
        [Input('ops_option_memory', 'data')],
        [State(col,'value'),State(col,'options')])
    def update_default_labels(data,old_label,old_options,k=col):
        # ctx = dash.callback_context

        # print(ctx.triggered[0]['prop_id'])
        if data:
            label = data[k]
            print(label)
            if label=='None':
                return old_options[0]['value']
            else:
                return label
        else:
            return old_label

####################################################################################################################################




