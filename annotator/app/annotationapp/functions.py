import pandas as pd
import datetime
import numpy as np
# from scipy.interpolate import interp1d
# import geopandas as gpd
# from shapely.geometry import Point
import copy
def datestring_to_timestamp(string):
    year,month,day=string.split('-')
    day=day.split('T')[0]
    date=pd.Timestamp(year = int(year), month = int(month), day = int(day), hour = 0, minute =0,second = 0)
    return date

def get_ms_datetime(df):
    return df.datetime +pd.to_timedelta(df.millisecond.round(-2).values,unit='ms')

def get_exact_datetime(df):
    return df.datetime +pd.to_timedelta(df.millisecond.values,unit='ms')

def smoothen_all_ldn_segments(gpses):
    smoothened=pd.DataFrame()
    for gps in gpses:
        smoothened=smoothened.append(smoothen(gps))
    return smoothened
    
def smoothen(ldn_unsmooth):
    ldn=copy.deepcopy(ldn_unsmooth)
    ldn_len=len(ldn)
    ldn['datetime']=get_ms_datetime(ldn)
    ldn['is_interpolated']=False
    ldn.index=ldn.datetime
    ldn =ldn.resample('100L').asfreq()
    ldn.ldn_segment_id=ldn.ldn_segment_id.iloc[0]
    ldn.datetime=ldn.index
    ldn.millisecond=ldn.index.microsecond/1000
    ldn['t']=range(len(ldn))
    
    if ldn_len>1:
        track_no_repeat=ldn.dropna(subset=['longitude','latitude','altitude'])
        X=track_no_repeat['t'].values
        y=track_no_repeat['longitude'].values
        y2=track_no_repeat['latitude'].values
        y3=track_no_repeat['altitude'].values        
        #print (y,y2,y3)
        if ldn_len==2:
            bw=1
        elif ldn_len==3:
            bw=2
        else:
            bw=3
        s=interp1d(X,y,bw)    
        s2=interp1d(X,y2,bw)
        s3=interp1d(X,y3,bw)
            
    # =============================================================================
    #     time_s=gps.index
    #     time=gps.dropna(subset=['longitude','latitude','altitude']).index
    # =============================================================================
        xs = np.linspace(0, len(ldn)-1, len(ldn))

        ys = s(xs)        
        ys2 = s2(xs)
        ys3 = s3(xs)
    # =============================================================================
    #     fig1, ax1 = plt.subplots()
    #     ax1.plot(time_s, ys, 'r', lw=1)
    #     ax1.plot(time, y, 'kx',color='b',markersize=0.1)
    #     ax1.set_title('Longitude')
    # =============================================================================
    # =============================================================================
    #     
    #     fig2, ax2 = plt.subplots()
    #     ax2.plot(time_s, ys2, 'r', lw=1)
    #     ax2.plot(time, y2, 'kx',color='b',markersize=0.1)
    #     ax2.set_title('Latitude')
    # =============================================================================
        
    # =============================================================================
    #     
    #     fig3, ax3 = plt.subplots()
    #     ax3.plot(time_s, ys3*3.28084, 'r', lw=1)
    #     ax3.plot(time, y3*3.28084, 'kx',color='b',markersize=0.5)
    #     ax3.set_title('Altitude')
    #     ax3.set_ylabel('Altitude (ft)')
    # =============================================================================
        ldn['latitude']=ys2
        ldn['longitude']=ys
        ldn['altitude']=ys3
    idx=ldn[ldn.is_interpolated.isnull()].index
    ldn.loc[idx,'is_interpolated']=True
    ldn=ldn.drop('t',1)
    return ldn


def get_geodf_from_df(df):
    geo= [Point(xy) for xy in zip(df.longitude, df.latitude)]
    #gdf = df.drop(['Long', 'Lat'], axis=1)
    return gpd.GeoDataFrame(crs= {'init': 'epsg:4326'}, geometry=geo)

