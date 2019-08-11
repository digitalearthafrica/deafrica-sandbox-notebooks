import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from time import time
import numpy as np


# Change the bands (RGB) here if you want other false color combinations
def rgb(dataset,
        at_index = 0,
        bands = ['red', 'green', 'blue'],
        paint_on_mask = [],
        max_possible = 3500,
        width = 10
       ):

    def aspect_ratio_helper(x,y, fixed_width = 20):
        width = fixed_width
        height = y * (fixed_width / x)
        return (width, height)
    
    ### < Dataset to RGB Format, needs float values between 0-1 
    rgb = np.stack([dataset[bands[0]],
                    dataset[bands[1]],
                    dataset[bands[2]]], axis = -1).astype(np.int16)
    
    rgb[rgb<0] = 0    
    rgb[rgb > max_possible] = max_possible # Filter out saturation points at arbitrarily defined max_possible value
    
    rgb = rgb.astype(float)
    rgb *= 1 / np.max(rgb)
    ### > 
    
    ### < takes a T/F mask, apply a color to T areas  
    for mask, color in paint_on_mask:        
        rgb[mask] = np.array(color)/ 255.0
    ### > 
    
    
    fig, ax = plt.subplots(figsize = aspect_ratio_helper(*rgb.shape[:2], fixed_width = width))

    lat_formatter = FuncFormatter(lambda x, pos: round(dataset.latitude.values[pos] ,4) )
    lon_formatter = FuncFormatter(lambda x, pos: round(dataset.longitude.values[pos],4) )

    plt.ylabel("Latitude")
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.xlabel("Longitude")
    ax.xaxis.set_major_formatter(lon_formatter)
   
    if 'time' in dataset:
        plt.imshow((rgb[at_index]))
    else:
        plt.imshow(rgb)  
    
    plt.show()