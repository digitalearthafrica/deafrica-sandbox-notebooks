from bokeh.models.widgets import Panel, Tabs
import numpy as np
import importlib
import pandas as pd
from bokeh.models import FuncTickFormatter
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.models.sources import ColumnDataSource
from bokeh.models import HoverTool


def __stem_figure(list1,
                name1,
                list2,
                name2,
                **kwargs):
    
    dataframe_A = pd.DataFrame(list1, index = list1).rename(columns = {0:name1})
    dataFrame_B = pd.DataFrame(list2, index = list2).rename(columns = {0:name2})
    


    result = pd.concat([dataframe_A,dataFrame_B], join = 'outer', axis = 1)
    result = (~pd.isnull(result)).astype(int)
    
    aframe = result[name1][ result[name1] > 0]
    bframe = result[name2][ result[name2] > 0]
    bframe = bframe.map(np.negative)    
    
    positive_match = aframe[result[name1] == result[name2]]
    positive_miss  = aframe[result[name1] != result[name2]]
    
    negative_match = bframe[result[name1] == result[name2]]
    negative_miss  = bframe[result[name1] != result[name2]]
        
    
    ## Handle DateTime formats on X-Axis
    x_axis_format = 'auto' if type(list1[0]) != np.datetime64 else "datetime" 

    ## Create Canvas/Figure 
    p = figure(plot_width  = 900,
               plot_height = 200,
               tools       = "xpan, reset, save, xzoom_in, xzoom_out",
               x_axis_type = x_axis_format,
               title= "Coordinate Comparison: {}".format(kwargs['dimension']))
        
    ## Create a long horizontal line
    start = min(aframe.index.min(), bframe.index.min())
    end   = max(aframe.index.max(), bframe.index.max())
    
    
    categories = ["oof", "doof"]

    
    ## Remove Horizontal Gridlines    
    p.ygrid.grid_line_color = None
    
    ## Remove YAxis Labels (Since they ploy categorical)
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
#     p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
        
    p.segment(x0 = start,
              x1 = end,
              y0 = 0,
              y1 = 0,
              line_width = 1)

    def __plot_stem_and_bulb(figure, series, color, name):  
    
        figure.circle(x = "index",
                      y = name,
                      source = ColumnDataSource(series.to_frame()),
                      alpha=0.5,
                      size = 10,
                      color = color)
        
        figure.segment(x0 = 'index',
                       x1 = 'index',
                       y0 = 0,
                       y1 = name,
                       source = ColumnDataSource(series.to_frame()),
                       line_width = 1,
                       color = color)
        return figure
    
    p = __plot_stem_and_bulb(p, positive_match, "green", name1)
    p = __plot_stem_and_bulb(p, positive_miss, "red", name1)
    p = __plot_stem_and_bulb(p, negative_match, "green", name2)
    p = __plot_stem_and_bulb(p, negative_miss, "red", name2)    
    
    p.yaxis.axis_label = "{}  {}".format(name2, name1)
    
    return p
    print(result)

def init_notebook():
    from bokeh.plotting import figure, show
    from bokeh.io import output_notebook
    output_notebook()
    
def dim_alignement(dataset_1 = None,
                          name_1 = "dataset_1",
                          dataset_2 = None,
                          name_2 = "dataset_2",
                          ):
    xr1 = dataset_1
    xr2 = dataset_2  
    
    common_dims = set(xr1.dims).intersection(set(xr2.dims))
    
    empty_set = set()
    if common_dims == empty_set:
        raise Exception("datasets do not have any dims in common")
        
    display_tabs = []
    for dim in common_dims:
        fig =  __stem_figure(xr1[dim].values,
                           name_1,
                           xr2[dim].values,
                           name_2,
                           dimension = dim
                        )
        
        display_tabs.append(Panel(child = fig, title = dim))
                
    tabs = Tabs(tabs = display_tabs) ## Make a figure with many tabs. 
    show(tabs)
            