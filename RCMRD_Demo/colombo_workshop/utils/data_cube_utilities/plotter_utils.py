import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import datacube as dc
import xarray as xr
import utils.data_cube_utilities.data_access_api as dc_api 
from utils.data_cube_utilities.dc_utilities import perform_timeseries_analysis
from utils.data_cube_utilities.dc_mosaic import ls7_unpack_qa
from rasterstats import zonal_stats
from scipy import stats
from scipy.stats import norm
import pylab
import matplotlib as mpl
from scipy.signal import gaussian
from scipy.ndimage import filters
from sklearn import linear_model
from scipy.interpolate import spline
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import calendar, datetime, time
import pytz

def n64_to_epoch(timestamp):
    ts = pd.to_datetime(str(timestamp)) 
    ts = ts.strftime('%Y-%m-%d')
    tz_UTC = pytz.timezone('UTC')
    time_format = "%Y-%m-%d"
    naive_timestamp = datetime.datetime.strptime(ts, time_format)
    aware_timestamp = tz_UTC.localize(naive_timestamp)
    epoch = aware_timestamp.strftime("%s")
    return (int) (epoch)

def regression_massage(ds): 
    t_len = len(ds["time"])
    s_len = len(ds["latitude"]) * len(ds["longitude"])
    flat_values = ds.values.reshape(t_len * s_len)
    return list(zip(list(map(n64_to_epoch, ds.time.values)),flat_values))

def remove_nans(aList):
    i = 0
    while i < len(aList):
        if np.isnan(aList[i][1]):
            del aList[i]
            i = 0
        else:
            i+=1
    return aList

def full_linear_regression(ds):
    myList = regression_massage(ds)
    myList = remove_nans(myList)
    myList = sorted(myList, key=lambda tup: tup[0])
    time, value = zip(*myList)
    value = [int(x) for x in value]
    value = np.array(value)
    value.astype(int)
    time = np.array(time)
    time.astype(int)
    return list(zip(time,value))
    
def tfmt(x, pos=None):
    return time.strftime("%Y-%m-%d",time.gmtime(x))

def plot_band(landsat_dataset, dataset):
    
    #Calculations
    times = dataset.time.values
    times = list(map(n64_to_epoch, times))
    times = np.array(times)
    times = np.sort(times)
    mean  = dataset.mean(dim=['latitude','longitude'],  skipna = True).values
    medians = dataset.median(dim=['latitude','longitude'], skipna = True)
    std_dev = np.nanstd(mean)
    plt.figure(figsize=(20,15))
    ax = plt.gca()

    #Shaded Area
    quarter = np.nanpercentile(
    dataset.values.reshape((
        landsat_dataset.dims['time'],
        landsat_dataset.dims['latitude'] * landsat_dataset.dims['longitude'])),
        25,
        axis = 1
    )
    three_quarters = np.nanpercentile(
    dataset.values.reshape((
        landsat_dataset.dims['time'],
        landsat_dataset.dims['latitude'] * landsat_dataset.dims['longitude'])),
        75,
        axis = 1
    )
    np.array(quarter)
    np.array(three_quarters)
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    fillcolor1='gray'
    fillcolor2='brown'
    fillalpha=0.4
    plt.fill_between(times, mean, quarter,  interpolate=False, color=fillcolor1, alpha=fillalpha,label="25th")
    plt.fill_between(times, mean, three_quarters, interpolate=False, color=fillcolor1, alpha=fillalpha,label="75th")
        
    #Medians
    plt.plot(times,medians,color="black",marker="o",linestyle='None', label = "Medians")
    
    #Linear Regression (on everything)
    #Data formatted in a way for needed for Guassian and Linear Regression
    #regression_list = full_linear_regression(dataset)
    #formatted_time, value = zip(*regression_list)
    #formatted_time = np.array(formatted_time)
    
    #The Actual Plot
    plt.plot(times,mean,color="blue",label="Mean")

    #Linear Regression (on mean)
    m, b = np.polyfit(times, mean, 1)
    plt.plot(times, m*times + b, '-', color="red",label="linear regression of mean",linewidth = 3.0)

    #Gaussian Curve
    b = gaussian(len(times), std_dev)
    ga = filters.convolve1d(mean, b/b.sum(),mode="reflect")
    x_smooth = np.linspace(times.min(),times.max(), 200)
    y_smooth = spline(times, ga, x_smooth)
    plt.plot(x_smooth, y_smooth, '-',label="Gaussian Smoothed of mean", alpha=1, color='limegreen',linewidth = 3.0)
    
    
    #Formatting
    ax.grid(color='k', alpha=0.1, linestyle='-', linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(tfmt))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    plt.show

def plot_pixel_qa_value(dataset, platform, values_to_plot, bands = "pixel_qa", plot_max = False, plot_min = False):
    times = dataset.time.values
    mpl.style.use('seaborn')
    plt.figure(figsize=(20,15))
    quarters = []
    three_quarters = []
    percentiles = []
   
    for i,v in enumerate(values_to_plot):
        _xarray  = ls7_unpack_qa(dataset.pixel_qa, values_to_plot[i])
        y = _xarray.mean(dim= ['latitude', 'longitude'])
        times = dataset.time.values.astype(float)
        std_dev = np.std(y)
        std_dev = std_dev.values
        b = gaussian(len(times), std_dev)
        ga = filters.convolve1d(y, b/b.sum(),mode="reflect")
        ga=interpolate_gaps(ga, limit=3)
        plt.plot(times, ga, '-',label="Gaussian ", alpha=1, color='black')
        
        x_smooth = np.linspace(times.min(),times.max(), 200)
        y_smooth = spline(times, ga, x_smooth)
        plt.plot(x_smooth, y_smooth, '-',label="Gaussian Smoothed", alpha=1, color='cyan')
        
        for i, q in enumerate(_xarray):
            quarters.append(np.nanpercentile(_xarray, 25))
            three_quarters.append(np.nanpercentile(_xarray, 75))
            #print(q.values.mean())
        
        ax = plt.gca()
        ax.grid(color='lightgray', linestyle='-', linewidth=1)
        fillcolor='gray'
        fillalpha=0.4
        linecolor='gray'
        linealpha=0.6
        plt.fill_between(times, y, quarters,  interpolate=False, color=fillcolor, alpha=fillalpha)
        plt.fill_between(times, y, three_quarters, interpolate=False, color=fillcolor, alpha=fillalpha)
        plt.plot(times,quarters,color=linecolor , alpha=linealpha)
        plt.plot(times,three_quarters,color=linecolor, alpha=linealpha)
        
        medians = _xarray.median(dim=['latitude','longitude'])
        plt.scatter(times,medians,color='mediumpurple', label="medians", marker="D")
        
        m, b = np.polyfit(times, y, 1)
        plt.plot(times, m*times + b, '-', color="red",label="linear regression")
        plt.style.use('seaborn')
        
        plt.plot(times, y, marker="o")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(rotation=90)    