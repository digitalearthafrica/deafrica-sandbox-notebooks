import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import math
import collections


def _is_list(var):
    return isinstance(var, (list, tuple))


def display_at_time(cubes, time = None, color = [238,17,17], width = 3, name = None, on_pixel = True, mode = None,h=4,w=10):
	lop = []
	for x in cubes:
		if isinstance(x, tuple):
			#Overlays the not-nan values of the 'subject' on the 'canvas'.
			canvas  = x[0].sel(time = time, method = 'nearest')
			subject = x[1].sel(time = time, method = 'nearest')
			overlay = _overlayer(canvas, subject, color = color,on_pixel = on_pixel, mode = mode)
			lop.append(overlay)		
		else:
			selection = x.sel(time = time, method = 'nearest')
			lop.append(_to_image(selection))
	_display_list_of_plottables(lop, maxwidth = width, name = name,h = h, w = w)

def _to_image(cube,minval = 0, maxval = 2000, backfill = []):
	red, green, blue =  [np.array(cube.red.values), np.array(cube.green.values), np.array(cube.blue.values)]
	red   = np.copy(red)
	green = np.copy(green)
	blue  = np.copy(blue)
	red[red < minval]     = minval
	green[green < minval] = minval
	blue[blue < minval]   = minval
	
	red[red > maxval]     = maxval
	green[green > maxval] = maxval
	blue[blue > maxval]   = maxval
	
	red = red/maxval
	green = green/maxval
	blue = blue/maxval 
	
	red   = (abs(red -1)   * 255).astype(np.int16)
	green = (abs(green -1) * 255).astype(np.int16) 
	blue  = (abs(blue -1)  * 255).astype(np.int16)
	
	rgb = np.dstack([red,green,blue])
	_reversedim(rgb, k = 0)
	return rgb
			

def _encode(h,w,e):
	return str(h) + str(w) + str(e)	 

def _display_list_of_plottables(plotables, maxwidth = 3, name= 'figure', h=4,w=10):
	if _is_list(plotables):	
		height = math.ceil(len(plotables)/maxwidth)
		plt.figure(name,figsize=(w, h))
		for index, item in enumerate(plotables):
			plt.subplot(_encode(height, maxwidth,index + 1))
			plt.imshow(item)
		plt.show()
	else:
		raise Execption("YOU NEED TO PASS A LIST")


def _reversedim(M,k=0):
    idx = tuple((slice(None,None,-1) if ii == k else slice(None) 
            for ii in range(M.ndim)))
    return M[idx]


def _overlayer(canvas, overlay, band = 'red',color = [238,17,17], on_pixel = True, mode = None):
	subject = overlay[band]
	if on_pixel is True:
		subject_indices = np.copy(np.dstack(np.where(~np.isnan(subject.values)))[0])
	else: 
		subject_indices = np.copy(np.dstack(np.where(np.isnan(subject.values)))[0])	
	rgb_canvas = np.copy(_to_image(canvas))
	if mode is 'blend':
		for x in subject_indices:
			rgb_canvas[x[0]][x[1]][0] = ((255 - color[0]) + rgb_canvas[x[0]][x[1]][0])/2
			rgb_canvas[x[0]][x[1]][1] = ((255 - color[1]) + rgb_canvas[x[0]][x[1]][1])/2
			rgb_canvas[x[0]][x[1]][2] = ((255 - color[2]) + rgb_canvas[x[0]][x[1]][2])/2
		return rgb_canvas
	else:
		for x in subject_indices:
			rgb_canvas[x[0]][x[1]][0] = 255 - color[0]
			rgb_canvas[x[0]][x[1]][1] = 255 - color[1]
			rgb_canvas[x[0]][x[1]][2] = 255 - color[2]
		return rgb_canvas


def __is_iterable(value):
	return isinstance(value, collections.Iterable)

def _np64_to_datetime(dt64):
	ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
	return datetime.datetime.utcfromtimestamp(ts)


