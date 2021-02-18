#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:05:00 2019

@author: dh
"""

#%%
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from .functions import get_ms_datetime
import plotly.express as px
import pyproj
import plotly.figure_factory as ff
import numpy as np
import math
from . import config as cfg
from interpolate import interpolate
from database import database_functions
from datetime import datetime
import colorsys
import time
#%%



def generate_table_figure(df):
    trace = go.Table(
    
    header = dict(values = [['<b>{}</b>'.format(c)] for c in df.columns],line = dict(color = '#004A53'),fill = dict(color = '#004A53'),
                             align = 'center',font = dict(color = 'white', size = 20),height=40),
    cells = dict(values = [df[col] for col in df.columns],line = dict(color = '#004A53'),fill = dict(color = 'white'),align = 'center',
               font = dict(color = '#011B18', size = 18),height=30))
   
    figure={'data':[trace],
		        'layout': go.Layout(
		            paper_bgcolor = 'white',
		            plot_bgcolor = 'white',
		          font=dict(family='helvetica', 
		                             size=14, 
		                             color='#2C3330'),
             margin=go.layout.Margin(l=5, r=5, t=5, b=5,pad=1)
		            )
		        }
    return figure

def process_df_for_map(df):
    df = interpolate.get_smoothen_track(df)
    proj=pyproj.Proj("+proj=utm +zone=48 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    df['utm_x'],df['utm_y']=proj(df.longitude_smooth.values,df.latitude_smooth.values)
    df['dx']=df.utm_x.diff().shift(-1)
    df['dy']=df.utm_y.diff().shift(-1)
    df['speed']=np.sqrt(df.dx*df.dx+df.dy*df.dy)/cfg.ARROW_LEN
    df['dx']=df['dx']/df.speed
    df['dy']=df['dy']/df.speed
    angle=float(-25) #rotate the speed vector by this amount anti clockwhise
    arad=angle/180*math.pi
    df['ddx']=df.dx*math.cos(arad)-df.dy*math.sin(arad)
    df['ddy']=df.dx*math.sin(arad)+df.dy*math.cos(arad)

    angle=float(25)
    arad=angle/180*math.pi
    df['dddx']=df.dx*math.cos(arad)-df.dy*math.sin(arad)
    df['dddy']=df.dx*math.sin(arad)+df.dy*math.cos(arad)

    df['xnormal']=df['latitude']-df['ddy'] #minus because we want it to trail backwards
    df['ynormal']=df['longitude']-df['ddx']

    df['xnormal2']=df['latitude']-df['dddy']
    df['ynormal2']=df['longitude']-df['dddx']

    df=df.dropna()

    return df

def get_arrow_trace(df):
    arrow_color = _get_arrow_color(df)
    lx=[]
    ly=[]
    ac = []
    for i in range(0,len(df),cfg.ARROW_PLOT_INTERVAL):
        lx=np.append(lx,np.append(df.iloc[i][['xnormal2','latitude','xnormal']].values,[None]))
        ly=np.append(ly,np.append(df.iloc[i][['ynormal2','longitude','ynormal']].values,[None]))
        ac.append(arrow_color[i])
    return lx,ly,ac
       
def get_arrow_frame(k,df_frame,n_arrow):
    lats=[]
    lons=[]

    for i in range(n_arrow):

        lats=np.append(lats,np.append(df_frame.iloc[k+i][['xnormal2','latitude','xnormal']].values,[None]))
        lons=np.append(lons,np.append(df_frame.iloc[k+i][['ynormal2','longitude','ynormal']].values,[None]))

    return lats,lons
def generate_frames(df,is_offline):
    if is_offline:
        map_type = 'scattergeo'
    else:
        map_type = 'scattermapbox'

    time=(df.iloc[-1].datetime-df.iloc[0].datetime).total_seconds()/60 #time in minutes
    N=np.min([20,time//cfg.ARROW_INTERVAL])
    N = len(df)-cfg.NUM_ARROWS 
    moving_index=[int(j) for j in np.linspace(0,len(df)-1-cfg.NUM_ARROWS,int(N))]
    frames=[]
    timestamps=[]
    for k in range(len(moving_index)):
        i=moving_index[k]
        lats,lons=get_arrow_frame(i,df,cfg.NUM_ARROWS)
        frames.append(dict(data= [dict(type=map_type,
                        lat=lats,
                        lon=lons)],
            traces= [2],
            name='frame{}'.format(k)       
            ) 
        )
        timestamps.append(str(df.iloc[i]['datetime']).split(' ')[1])
    return frames,timestamps
def get_map_center(lats,lons):
    #1.3521,103.8198
    return lats.mean(),lons.mean()

def generate_track_color(df):
    colorlist = []
    colors = {
        'full': '#EF476F',
        'callsign': '#086788',
        'squad_code': '#FF9F1C',
        'no_stitch':'#00CFC1',
        'double': '#EB5E28'
        }
    # clist = ['#EF476F','#00CFC1','#086788','#FF9F1C','#EB5E28']
    # track_id =df.source_track_id.unique()
    # colors = {track_id[i]: clist[i%5] for i in range(len(track_id))}

    for i in range(len(df)):
        colorlist.append(colors[df.iloc[i].stitch_type])
        # colorlist.append(colors[df.iloc[i].source_track_id])
    return colorlist

def generate_cord_figure(df,click_data,old_state,cord): #for showing the 3 graphs at the side
    '''
    @param: df: 1 row represents a point
            click_data: for showing the blue vertical data in the graphs (don't care)
            old_state: dont'care
            cord: latitude, longitude or altitude
    '''
    df = df.copy()
    if cord == 'altitude':
        df.loc[:,'altitude'] = df.altitude* 3.28084/100
        label = 'Altitude (100 Feet)'
    elif cord =='latitude':
        label = 'Latitude'
    elif cord =='longitude':
        label = 'Longitude'
    try:
        #df.sort_values(by = ['datetime'])
        cats = df['Patterns'].unique() #for predictions
        x = list(range(0, df.shape[0]))
    except Exception:
        df = df.sort_values(by = ['datetime']) #for real data
        cats = df['source_track_id'].unique()
        x = df.datetime.dt.time
        #print(df.datetime.dt.time.values.tolist())
    try:
        uniquetrackids = df['track_id'].unique() #for real data
    except Exception:
        uniquetrackids = df['TrajectoryId'].unique() #for predictions
    #colours = get_main_hue_colour(len(uniquetrackids))
    #print(colours)
    data = []
    for i in range(len(uniquetrackids)):
        track = uniquetrackids[i]
        try:
            sub_df = df[df['track_id'] == track] #for real data
            x1 = sub_df.datetime.dt.time
        except Exception:
            sub_df = df[df['TrajectoryId'] == track] #for predictions
            x1 = list(range(0, sub_df.shape[0]))
        if i > 0:
            data.append( #hide legend
                go.Scatter(
                    mode='lines+markers',
                    y=sub_df[cord],
                    x = x1,
                    showlegend=False,
                    marker = dict(
                    size = 2,
                    symbol='circle-open',
                    color = '#00FF00' 
                    )
                )
                )
        else:
            data.append(
                go.Scatter( #only show legend for the first one
                    mode='lines+markers',
                    y=sub_df[cord],
                    x = x1,
                    name=label,
                    marker = dict(
                    size = 2,
                    symbol='circle-open',
                    color = '#00FF00' 
                    )
                )
                )
    # data = [
    #         go.Scatter(
    #             mode='lines',
    #             y=df[cord],
    #             x = x,
    #             name=label,
    #             color = '#00FF00' 
    #         ),
    #         ]
    if cord == 'altitude':
        for i in range(len(uniquetrackids)):
            track = uniquetrackids[i]
            try:
                sub_df = df[df['track_id'] == track] #for real data
                sub_df = sub_df.sort_values(by = ['datetime'])
                x1 = sub_df.datetime.dt.time
            except Exception:
                sub_df = df[df['TrajectoryId'] == track]
                x1 = list(range(0, sub_df.shape[0]))
            if i > 0:
                data.append(
                    go.Scatter( #hide legend
                        mode='lines+markers',
                        y=sub_df.speed,
                        x = x1,
                        showlegend=False,
                        marker = dict(
                        size = 2,
                        symbol='circle-open',
                        color = '#FF00FF'
                        )
                    )
                    )
            else:
                data.append(
                    go.Scatter( #only show legend for the first one
                        mode='lines+markers',
                        y=sub_df.speed,
                        x = x1,
                        name= 'Speed (knots)',
                        marker = dict(
                        size = 2,
                        symbol='circle-open',
                        color = '#FF00FF'
                        )
                    )
                    )
        # data.append(go.Scatter(
        #         mode='lines',
        #         y=df.speed,
        #         x = x,
        #         name='Speed (knots)',
        #         color = '#FF00FF'
        #     ))
    # if len(cats) == 1:
    #     layout = {
    #         'xaxis_range' : [-1, len(x) + 1],
    #         'shapes': [
    #             {
    #                 'type': 'rect',
    #                 # x-reference is assigned to the x-values
    #                 'xref': 'x',
    #                 # y-reference is assigned to the plot paper [0,1]
    #                 'yref': 'paper',
    #                 'x0': '-1',
    #                 'y0': 0,
    #                 'x1': str(len(x) + 1),
    #                 'y1': 1,
    #                 'fillcolor': get_main_map_colour(0),
    #                 'opacity': 0.2,
    #                 'line': {
    #                     'width': 0,
    #                 }
    #             }
    #         ]
    #     }
    #     fig = go.Figure(data = data, layout = layout)
    # elif len(cats) == 2:
    #     try:
    #         idx = df['Patterns'].tolist().index(cats[1])
    #     except Exception:
    #         idx = df['source_track_id'].tolist().index(cats[1])
    #     layout = {
    #         'xaxis_range' : [-1, len(x) + 1],
    #         'shapes': [
    #             {
    #                 'type': 'rect',
    #                 # x-reference is assigned to the x-values
    #                 'xref': 'x',
    #                 # y-reference is assigned to the plot paper [0,1]
    #                 'yref': 'paper',
    #                 'x0': '-1',
    #                 'y0': 0,
    #                 'x1': str(idx),
    #                 'y1': 1,
    #                 'fillcolor': get_main_map_colour(0),
    #                 'opacity': 0.2,
    #                 'line': {
    #                     'width': 0,
    #                 }
    #             },
    #             {
    #                 'type': 'rect',
    #                 # x-reference is assigned to the x-values
    #                 'xref': 'x',
    #                 # y-reference is assigned to the plot paper [0,1]
    #                 'yref': 'paper',
    #                 'x0': str(idx),
    #                 'y0': 0,
    #                 'x1': str(len(x) + 1),
    #                 'y1': 1,
    #                 'fillcolor': get_main_map_colour(1),
    #                 'opacity': 0.2,
    #                 'line': {
    #                     'width': 0,
    #                 }
    #             },
    #         ]
    #     }
    #     fig = go.Figure(data = data, layout = layout)
    # elif len(cats) == 3:
    #     try:
    #         idx = df['Patterns'].tolist().index(cats[1])
    #         idx2 = df['Patterns'].tolist().index(cats[2])
    #     except Exception:
    #         idx = df['source_track_id'].tolist().index(cats[1])
    #         idx2 = df['source_track_id'].tolist().index(cats[2])
    #     layout = {
    #         'xaxis_range' : [-1, len(x) + 1],
    #         'shapes': [
    #             {
    #                 'type': 'rect',
    #                 # x-reference is assigned to the x-values
    #                 'xref': 'x',
    #                 # y-reference is assigned to the plot paper [0,1]
    #                 'yref': 'paper',
    #                 'x0': '-1',
    #                 'y0': 0,
    #                 'x1': str(idx),
    #                 'y1': 1,
    #                 'fillcolor': get_main_map_colour(0),
    #                 'opacity': 0.2,
    #                 'line': {
    #                     'width': 0,
    #                 }
    #             },
    #             {
    #                 'type': 'rect',
    #                 # x-reference is assigned to the x-values
    #                 'xref': 'x',
    #                 # y-reference is assigned to the plot paper [0,1]
    #                 'yref': 'paper',
    #                 'x0': str(idx),
    #                 'y0': 0,
    #                 'x1': str(idx2),
    #                 'y1': 1,
    #                 'fillcolor': get_main_map_colour(1),
    #                 'opacity': 0.2,
    #                 'line': {
    #                     'width': 0,
    #                 }
    #             },
    #             {
    #                 'type': 'rect',
    #                 # x-reference is assigned to the x-values
    #                 'xref': 'x',
    #                 # y-reference is assigned to the plot paper [0,1]
    #                 'yref': 'paper',
    #                 'x0': str(idx2),
    #                 'y0': 0,
    #                 'x1': str(len(x) + 1),
    #                 'y1': 1,
    #                 'fillcolor': get_main_map_colour(2),
    #                 'opacity': 0.2,
    #                 'line': {
    #                     'width': 0,
    #                 }
    #             }
    #         ]
    #     }
    #     fig = go.Figure(data = data, layout = layout)
    #else:
    layout = {
    }
    fig = go.Figure(data = data, layout = layout)

    '''
    if len(cats) == 1:
        first_idx = 0 
    elif len(cats) == 2:
        try:
            first_idx = df['Patterns'].tolist().index(cats[1])
        except Exception:
            first_idx = df['source_track_id'].tolist().index(cats[1])
    else:
        try:
            first_idx = df['Patterns'].tolist().index(cats[1])
            second_idx = df['Patterns'].tolist().index(cats[2])
        except Exception:
            first_idx = df['source_track_id'].tolist().index(cats[1])
            second_idx = df['source_track_id'].tolist().index(cats[2])
    
    if click_data:
        if click_data['points'][0]['curveNumber'] == 1:
            idx = click_data['points'][0]['pointNumber'] + first_idx
        elif click_data['points'][0]['curveNumber'] == 0:
            idx = click_data['points'][0]['pointNumber']
        elif click_data['points'][0]['curveNumber'] == 2:
            idx = click_data['points'][0]['pointNumber'] + first_idx + second_idx

        #print(df.iloc[idx, ])
        x = idx
        y_max = df[cord].max()
        y_min = df[cord].min()
        if cord == 'altitude':
            y_max = max(df.speed.max(),y_max)
            y_min = min(df.speed.min(),y_min)
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=x,
                y0=y_min,
                x1=x,
                y1=y_max,
                line=dict(
                    color="RoyalBlue",
                    width=1
                )
        ))

        #elif old_state:
            #return old_state
    '''
    return fig
def _alt_2_group(alt):
    if alt<=30:
        return 1
    elif alt<=50:
        return 2
    elif alt<=100:
        return 3
    elif alt<=200:
        return 4
    else:
        return 5
def _get_arrow_color(df):
    color_map = {
            1: '#EF476F',
            2:'#00CFC1',
            3: '#086788',
            4: '#FF9F1C',
            5: '#EB5E28'
        }
    arrow_color = df.altitude.apply(_alt_2_group).map(color_map).tolist()
    return arrow_color

def get_main_map_colour(idx): #get colours
    '''
    @param: idx: an integer
    '''
    color_map = {0: '#0000FF', 1: '#FF0000', 2:'#086788'}
    return color_map[idx]

def get_main_hue_colour(num): #get colours based on the number of unique ids
    '''
    @param: num: number of unique ids
    '''
    N = num
    step = 1/N
    HSV_tuples = [(x * step, 1, 1) for x in range(N)]
    #print(HSV_tuples)
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    #print(RGB_tuples)
    tmp = []
    for i in RGB_tuples:
        inter = ()
        for j in i:
            inter += (round(j * 255),)
        tmp.append(inter)
    RGB_tuples = tmp
    #print(RGB_tuples)
    RGB_tuples = ["#%02x%02x%02x" % i for i in RGB_tuples]
    #print(RGB_tuples)
    return RGB_tuples

def generate_map_figure(df,is_offline,map_options): #for showing the main map
    '''
    @param: df: 1 row represents a point
            is_offline: boolean (true or false)
            map_options: show directional arrows or not, will be a list
    '''
    show_animation = False
    fig = go.Figure()
    try:
        df.loc[:, 'datetime'] = df.timestamp #for predictions
    except Exception:
        df = df
    #print(df.shape)
    #df=process_df_for_map(df).copy()
    #print(df.shape)
    df.loc[:,'altitude'] = df.altitude* 3.28084/100

    map_lat,map_lon=get_map_center(df.latitude,df.longitude)
    #df_interpolated = interpolate.get_interpolation(df)
    '''
    try: 
        df=process_df_for_map(df).copy()
        df.loc[:,'altitude'] = df.altitude* 3.28084/100

        map_lat,map_lon=get_map_center(df.latitude,df.longitude)
        df_interpolated = interpolate.get_interpolation(df)

    except Exception:
        df.loc[:,'altitude'] = df.altitude* 3.28084/100
        map_lat,map_lon=get_map_center(df.latitude,df.longitude)
    '''

    if show_animation:
        frames,timestamps =generate_frames(df,is_offline) 
        sliders = [dict(steps= [dict(method= 'animate',
                           args= [[ 'frame{}'.format(k) ],
                                  dict(mode= 'immediate',
                                  frame= dict( duration=100, redraw= True ),
                                           transition=dict( duration= 0)
                                          )
                                    ],
                            label='{}'.format(timestamps[k])
                             ) for k in range(len(frames))], 
                transition= dict(duration= 0 ),
                x=0,#slider starting position  
                y=0, 
                currentvalue=dict(font=dict(size=12), 
                                  prefix='Time: ', 
                                  visible=True, 
                                  xanchor= 'center'),  
                len=1.0)
           ]
    if is_offline:
        if 'TrajectoryId' in list(df.columns.values):
            uniquepatterns = df['Patterns'].unique()
            if len(uniquepatterns) == 1: #hard-coding here
                fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'Patterns', color_discrete_sequence = [get_main_map_colour(0)], opacity = 0.2)
            elif len(uniquepatterns) == 2:
                fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'Patterns', color_discrete_sequence = [get_main_map_colour(0), get_main_map_colour(1)], opacity = 0.2)
            else:
                fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'Patterns', color_discrete_sequence = [get_main_map_colour(0), get_main_map_colour(1), get_main_map_colour(2)], opacity = 0.2)
        else:
            uniquepatterns = df['source_track_id'].unique()
            df["source_track_id"] = df["source_track_id"].astype(str) #convert to string or else it won't follow the colours
            if len(uniquepatterns) == 1:
                fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'source_track_id', color_discrete_sequence = [get_main_map_colour(0)], opacity = 0.2)
            elif len(uniquepatterns) == 2:
                fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'source_track_id', color_discrete_sequence = [get_main_map_colour(0), get_main_map_colour(1)], opacity = 0.2)
            else:
                fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'source_track_id', color_discrete_sequence = [get_main_map_colour(0), get_main_map_colour(1), get_main_map_colour(2)], opacity = 0.2)
        fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=1, mapbox_center_lat = 1.40, mapbox_center_lon = 104,
        margin={"r":0,"t":30,"l":0,"b":0})
        
        #else:
        '''
        fig.add_trace(go.Scattergeo(
            lat=df_interpolated['latitude'],
            lon=df_interpolated['longitude'],
            mode='markers',
            marker=dict(
                size=cfg.MARKER_SIZE,
                color='#000000',#'#D31996',
                opacity=0.7
            ),
            text='interpolated',
            hoverinfo='text'
        ))   
        fig.add_trace(go.Scattergeo(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',
            marker=dict(
                size=cfg.MARKER_SIZE,
                color='#D31996',
                opacity=0.1
            ),
            # text=df['stitch_type'],
            hoverinfo='text'
        ))
        '''

        # fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="TrajectoryId", zoom=3)
        # fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=1, mapbox_center_lat = 1.40, mapbox_center_lon = 104,
        # margin={"r":0,"t":30,"l":0,"b":0})
        # return fig

        if 'show_arrow' in map_options: #putting arrows on the map
            lx,ly,arrow_color=get_arrow_trace(df)
            for j in range(len(lx)//4):
                fig.add_trace(go.Scattergeo(
                lat=lx[(j*4):(j+1)*4],
                lon=ly[(j*4):(j+1)*4],
                mode='lines',
                line=dict(
                    width=cfg.ARROW_WIDTH,
                    color=arrow_color[j]#,
                
                ),
                hoverinfo='skip',

            ))
        return fig
       
        fig.update_layout(
            height=900, 
            geo = dict(
                # framewidth = 1,
                showland = True, showlakes = False, showocean=False, showcountries=False,
                landcolor = 'rgb(204, 204, 204)',
                oceancolor= 'rgb(145, 191, 219)',
                countrycolor = 'rgb(128, 128, 128)',
                lakecolor = 'rgb(145, 191, 219)',
                countrywidth = 0.5,
                subunitwidth = 0.5,
                coastlinewidth = 1,
                resolution = 50,#50 or 110, 50 is better,
                center = dict(lat=map_lat,lon=map_lon),
                projection = go.layout.geo.Projection(
                # type = 'azimuthal equal area',
                scale=40
            )
            ),
            margin={"r":0,"t":0,"l":0,"b":0},)
    
    else:
        '''
        fig.add_trace(go.Scattermapbox(
                lat=df_interpolated['latitude'],
                lon=df_interpolated['longitude'],
                mode='markers',
                # name='inter',
                marker=go.scattermapbox.Marker(
                    size=cfg.MARKER_SIZE,
                    color='#000000',
                    opacity=0.7,
                    # symbol = 'triangle'
                ),
                text='interpolated',
                hoverinfo = 'text'
            ))

        #this trace is the actual points
        fig.add_trace(go.Scattermapbox(
                lat=df['latitude'],
                lon=df['longitude'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=cfg.MARKER_SIZE,
                    color='#D31996',
                    opacity=0.7
                ),
                text=df['datetime'].dt.time,
                hoverinfo='text'
            ))
        '''
        if 'TrajectoryId' in list(df.columns.values): #for predictions
            #uniquepatterns = df['Patterns'].unique()
            # if len(uniquepatterns) == 1:
            #     fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'Patterns', color_discrete_sequence = [get_main_map_colour(0)], opacity = 0.2)
            # elif len(uniquepatterns) == 2:
            #     fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'Patterns', color_discrete_sequence = [get_main_map_colour(0), get_main_map_colour(1)], opacity = 0.2)
            # else:
            #     fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'Patterns', color_discrete_sequence = [get_main_map_colour(0), get_main_map_colour(1), get_main_map_colour(2)], opacity = 0.2)
            df["TrajectoryId"] = df["TrajectoryId"].astype(str)
            numuniquetrackids = len(df.TrajectoryId.unique())
            fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'TrajectoryId', color_discrete_sequence = get_main_hue_colour(numuniquetrackids))

        else: #for real data
            #uniquepatterns = df['source_track_id'].unique()
            start_time = time.time()
            df["track_id"] = df["track_id"].astype(str)
            numuniquetrackids = len(df.track_id.unique())
            uniquetrackids = df["track_id"].unique()
            #fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=3, color = 'track_id', color_discrete_sequence = get_main_hue_colour(numuniquetrackids))
            fig = go.Figure()

            #numuniquelabels = len(df.new_label.unique())
            #uniquelabels = df['new_label'].unique()
            colors = get_main_hue_colour(numuniquetrackids) #retrieve the colors which can change based on how many unique ids there are
            
            for i in range(len(uniquetrackids)):
                track_id = uniquetrackids[i]
                sub_df = df[df['track_id'] == track_id]
                if 'Actual' in track_id:
                    fig.add_trace(go.Scattermapbox( #opacity 1 for real data
                        lat=sub_df['latitude'],
                        lon=sub_df['longitude'],
                        mode='markers',
                        name = track_id,
                        marker = dict(
                            color = colors[i]),
                        text=sub_df['track_id'],
                        hoverinfo='text'
                    ))
                else:
                    fig.add_trace(go.Scattermapbox( #opacity 0.5 for predictions
                        lat=sub_df['latitude'],
                        lon=sub_df['longitude'],
                        mode='markers',
                        name = track_id,
                        marker = dict(
                            color = colors[i], opacity = 0.5),
                        text=sub_df['track_id'],
                        hoverinfo='text'
                    ))
            
            '''
            for i in range(len(uniquelabels)):
                label = uniquelabels[i]
                sub_df = df[df['new_label'] == label]

                fig.add_trace(go.Scattermapbox(
                    lat=sub_df['latitude'],
                    lon=sub_df['longitude'],
                    mode='markers',
                    name = label,
                    marker = dict(
                        color = colors[i]),
                    text=sub_df['new_label'],
                    hoverinfo='text'
                ))
            '''
            '''
            for i in range(len(uniquetrackids)):
                trackid = uniquetrackids[i]
                sub_df = df[df['track_id'] == trackid]

                fig.add_trace(go.Scattermapbox(
                    lat=sub_df['latitude'],
                    lon=sub_df['longitude'],
                    mode='markers',
                    name = trackid,
                    marker = dict(
                        color = colors[i]),
                    text=sub_df['track_id'],
                    hoverinfo='text'
                ))
            '''
        fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=1, mapbox_center_lat = map_lat, mapbox_center_lon = map_lon, margin={"r":0,"t":0,"l":0,"b":0}, showlegend = True)
        #fig.update_layout(height=900,mapbox_style='carto-positron', mapbox_zoom=6, mapbox_center_lat = map_lat, mapbox_center_lon=map_lon,
        #margin={"r":0,"t":0,"l":0,"b":0},showlegend=True)

        '''
        #this trace is the arrows
        if 'show_arrow' in map_options:
            lx,ly,arrow_color=get_arrow_trace(df)
            for j in range(len(lx)//4):
                fig.add_trace(go.Scattermapbox(
                        lat=lx[(j*4):(j+1)*4],
                        lon=ly[(j*4):(j+1)*4],
                        mode='lines',
                        line=go.scattermapbox.Line(
                            width=cfg.ARROW_WIDTH,
                            color=arrow_color[j],#'#D31996',
                        
                        ),
                        hoverinfo='skip',

                    ))
        '''

        #fig.update_layout(height=900,mapbox_style='carto-positron', mapbox_zoom=6, mapbox_center_lat = map_lat, mapbox_center_lon=map_lon,
        #margin={"r":0,"t":0,"l":0,"b":0},showlegend=True)

    if show_animation:
        fig.update_layout(showlegend=False,updatemenus=[dict(type='buttons', showactive=False,
                        y=0,
                        x=1.05,
                        xanchor='right',
                        yanchor='top',
                        pad=dict(t=0, r=10,b=100),
                        buttons=[dict(label='Play',
                                        method='animate',
                                        args=[None, 
                                            dict(frame=dict(duration=100, 
                                                            redraw=True),
                                                    transition=dict(duration=0),
                                                    fromcurrent=True,
                                                    mode='immediate'
                                                )
                                            ]
                                        )
                                ]
                        )
                    ],
        sliders=sliders
        )
    else:
        fig.update_layout(showlegend=True)
    #end_time = time.time()
    #tot_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)) 
    #print(tot_time)
    return fig

