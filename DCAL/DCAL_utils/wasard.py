import datacube
import xarray as xr
import datetime
import warnings; warnings.simplefilter('ignore')
import numpy as np
from datetime import datetime
from time import time 
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage 
from sklearn.externals import joblib
from dc_water_classifier import wofs_classify
import random
import itertools
from sklearn.metrics import f1_score, recall_score, precision_score
dc = datacube.Datacube(app = 'wasard_test', config = '/home/localuser/.datacube.conf')


class wasard_classifier:
    """Classifier object used for classifying water bodies in a SAR dataset
    :attribute classifier: LinearSVC object used in classifying datapoints as water or not water
    :attribute coefficient: coefficient of the LinearSVC classifier, used to determine what data will classified as water and not water
    :attribute precision: classifier's precision score, using the Landsat scene it was trained on as truth labels
    :attribute recall: classifier's recall score, using the Landsat scene it was trained on as truth labels
    :attribute f1: classifier's precision score, created by taking the harmonic mean of the recall and precision scores
    :return: wasard_classifier object
    """
    def __init__(self, 
                 classifier         = None,
                 sar_dataset        = None, 
                 landsat_dataset    = None, 
                 pct                = .1, 
                 bands              = 2, 
                 sar_time_index     = -1, 
                 landsat_time_index = -1):
        """Defines the classifier as the loaded filestring passed in by the user, if a filestring passed in, otherwise trains a new classifier
        :param filestring: .pkl filestring of a previously trained classifier
        :param sar_dataset: xarray dataset containing sar data, loaded form datacube
        :param landsat_dataset: xarray dataset containing sar data, loaded form datacube
        :param pct: ratio of total training data to be used to train the classifier, lower numbers yield faster runtimes
        :param bands: indicates whether the classifier will be trained using 1 or 2 features
        :param sar_time_index: specific time index of sar scene to be used to train classifier, if the user wishes to specify
        :param sar_time_index: specific time index of landsat scene to be used to train classifier, if the user wishes to specify
        """
        if classifier:
            self.classifier  = joblib.load(classifier) if type(classifier) == str else classifier
            self.coefficient = self.classifier.coef_
        else:
            assert sar_dataset and landsat_dataset, "User must pass in either a .pkl filestring or both a SAR and Landsat dataset"
            self.train_classifier(
                                  sar_dataset        = sar_dataset, 
                                  landsat_dataset    = landsat_dataset, 
                                  pct                = pct, 
                                  bands              = bands, 
                                  sar_time_index     = sar_time_index, 
                                  landsat_time_index = landsat_time_index)
        
    def train_classifier(self, 
                         sar_dataset        = None, 
                         landsat_dataset    = None, 
                         pct                = .1, 
                         bands              = 2, 
                         sar_time_index     = -1, 
                         landsat_time_index = -1):
        """generates a classifier for a sar dataset using a support vector machine
        :param sar_dataset: xarray dataset containing sar data, loaded form datacube
        :param landsat_dataset: xarray dataset containing sar data, loaded form datacube
        :param pct: ratio of total training data to be used to train the classifier, lower numbers yield faster runtimes
        :param bands: indicates whether the classifier will be trained using 1 or 2 features
        :param sar_time_index: specific time index of sar scene to be used to train classifier, if the user wishes to specify
        :param sar_time_index: specific time index of landsat scene to be used to train classifier, if the user wishes to specify
        :return: LinearSVC object, which can be used to classify other sar scenes
        """
        
        train_data, test_data = _get_train_data(sar_dataset, 
                                                landsat_dataset, 
                                                pct                = pct, 
                                                bands              = bands, 
                                                sar_time_index     = sar_time_index, 
                                                landsat_time_index = landsat_time_index)
        
        # separate data attributes from target/truth value in order to train classifier with correct labels
        training_attributes  = train_data[:,0:-1]
        training_truth       = train_data[:,-1]
        testing_attributes   = test_data[:,0:-1]
        testing_truth        = test_data[:,-1]
        
        # train SVM using training data
        classifier           = svm.LinearSVC()
        classifier.fit(training_attributes, training_truth)

        # classify test dataset in order to verify its effectiveness
        total_tests          = testing_truth.size 
        testing_attributes   = np.stack(testing_attributes)
        wasard_predictions   = classifier.predict(testing_attributes)
        # print(wasard_predictions.shape)
        # wasard_predictions   = _filter_isolated_cells(wasard_predictions, struct=np.ones((3,3)), max_size = 200)
        
        # 3 metrics used to measure the effectiveness of the classifier
        f1                   = f1_score(testing_truth, wasard_predictions)
        recall               = recall_score(testing_truth, wasard_predictions)
        precision            = precision_score(testing_truth, wasard_predictions)
        
        sar_index, lsat_index = _find_training_indices(sar_dataset, landsat_dataset)

        sar_scene = sar_dataset.isel(time=sar_index).expand_dims('time')
        clf       = wasard_classifier(classifier=classifier)
        temp      = clf.wasard_classify(sar_scene)
    
        f1, precision, recall = _get_scores(temp, landsat_dataset, lsat_index)
        self.f1              = f1
        self.recall          = recall
        self.precision       = precision
        self.classifier      = classifier
        self.coefficient     = self.classifier.coef_
   
   
    def wasard_classify (self, sar_dataset):
        """Return new xarray Dataset identical to sar_dataset but with predicted water values added, using a provided classifier
        :param sar_dataset: xarray Dataset containing sar data, loaded from datacube
        :return: new xarray Dataset identical to sar_dataset with new array "wasard" added, containing predicted water values
        """
        satellite_type      = 'sentinel' if hasattr(sar_dataset, 'vv') else 'alos'
        dataset_dims        = (sar_dataset.coords.dims['time'], sar_dataset.coords.dims['latitude'], sar_dataset.coords.dims['longitude'])
        
        bands           = self.coefficient.size
        
        band2 = None
        if satellite_type   == 'sentinel':
            band1       = sar_dataset.vh.values.ravel()
            if bands == 2:
                band2       = sar_dataset.vv.values.ravel()
        elif satellite_type == 'alos':
            band1       = sar_dataset.hv.values.ravel()
            # band2       = sar_dataset.incidence_angle.values.ravel()

        # assemble features into array and predict water using the classifier
        if bands == 1:
            dinput      = np.dstack([band1])
            predictions = self.classifier.predict(dinput[0])
        elif bands == 2:
            dinput      = np.dstack([band1, band2])
            predictions = self.classifier.predict(dinput[0])
        else:
            raise ValueError("Bands must be 1 or 2")

        predictions                = predictions.reshape(dataset_dims)
        sar_dataset_copy           = sar_dataset.copy(deep=True)
        sar_dataset_copy['wasard'] = (('time', 'latitude', 'longitude'),predictions)

        # reduce noise from speckle and other isolated fals positives
        sar_dataset_copy           = _filter_all(sar_dataset_copy, max_size=100)

        return sar_dataset_copy

    def save(self, filestring):
        """saves a classfier to the disk
        :param filestring: name of the file which will contain the classifier
        """
        joblib.dump(self.classifier, "{}.pkl".format(filestring))

def get_best_classifier( n_classifiers,
                         sar_dataset        = None, 
                         landsat_dataset    = None, 
                         pct                = .1, 
                         bands              = 2, 
                         sar_time_index     = -1, 
                         landsat_time_index = -1):
    """generates a list of classifiers for a sar dataset using a support vector machine
        :param n: int indicating the number of classifiers that should be included in the returned list 
        :param sar_dataset: xarray dataset containing sar data, loaded form datacube
        :param landsat_dataset: xarray dataset containing sar data, loaded form datacube
        :param pct: ratio of total training data to be used to train the classifier, lower numbers yield faster runtimes
        :param bands: indicates whether the classifier will be trained using 1 or 2 features
        :param sar_time_index: specific time index of sar scene to be used to train classifier, if the user wishes to specify
        :param sar_time_index: specific time index of landsat scene to be used to train classifier, if the user wishes to specify
        :return: list containing n classifiers, which the user can then sort to find the most effective one
        """
    clf_ls = [wasard_classifier(  sar_dataset        = sar_dataset, 
                                  landsat_dataset    = landsat_dataset, 
                                  pct                = pct, 
                                  bands              = bands, 
                                  sar_time_index     = sar_time_index, 
                                  landsat_time_index = landsat_time_index) for x in range(n_classifiers)]
    
    clf_ls_precision_sorted = sorted(clf_ls, key = lambda clf: clf.precision, reverse=True)
    
    return clf_ls_precision_sorted

def wasard_plot(sar_dataset,
                sar_time_index,
                landsat_dataset=None,
                landsat_time_index=None,
                size=(10,10),
                plot_over_image = False):


    """Plots water values predicted from SAR data
    :param sar_dataset: xarray Dataset of sar data, with wasard values added from wasard_classify
    :param sar_time_index: int indicating the time slice to pull the wasard values from
    :param landsat_dataset: xarray containing data from landsat_dataset, with latitudinal and longitudinal bounds identical to the Sentinel scene_index, used for identifying watervalues detected by WASAR that do not correspond with those detected by WOFS
    :param landsat_time_index: time slice from the landsat_dataset array to be compared to the SAR scene_index
    :param size: tuple indicating the size of the output plot
    :param plot_over_image: boolean indicating whether the wasard values will be plot on top of a landsat_dataset image, preventing a new figure from being drawn
    :return: None
    """
    
    sar_dataset_at_time         = sar_dataset.isel(time=sar_time_index).copy(deep=True)
    # prevents function from setting up a new plot, if the data are to be plotted over an image
    if not plot_over_image: fig = plt.figure(figsize=size) 

    # plot wasard values over proper coordinates using xarray.dataarray.plot
    try: 
        water = sar_dataset_at_time.wasard.where(sar_dataset_at_time.wasard==1)
        water.plot.imshow(levels = 5, colors = 'blue', add_colorbar = 0, add_labels=0)
    # throws ValueError if wasard detected no water
    except ValueError: 
        pass
        
    if landsat_dataset:
        landsat_dataset_at_time       = landsat_dataset.isel(time=landsat_time_index)
        landsat_dataset_wofs          = get_wofs_values(landsat_dataset_at_time)
        wofs_resolution_adjusted      = _fit_landsat_dataset_resolution(landsat_dataset_wofs, sar_dataset)
        diff_array                    = (wofs_resolution_adjusted - sar_dataset_at_time.wasard).values
        sar_dataset_at_time['diff']   = (('latitude','longitude'),diff_array)
        sar_dataset_at_time['FalseN'] = sar_dataset_at_time['diff'].where(sar_dataset_at_time['diff'] == 1)
        false_neg_bool_array          = np.logical_and(sar_dataset_at_time.wasard==1,sar_dataset_at_time['diff'] != 1)
        sar_dataset_at_time['FalseP'] = sar_dataset_at_time['diff'].where(false_neg_bool_array)
        try:
            false_positives           = sar_dataset_at_time.FalseP
            false_negatives           = sar_dataset_at_time.FalseN
            false_positives.plot.imshow(levels = 5, colors = 'red', add_colorbar = 0, add_labels=0)
            false_negatives.plot.imshow(levels = 5, colors = 'yellow', add_colorbar = 0, add_labels=0)
        except ValueError: 
            pass


def wasard_time_plot(sar_dataset, size=(15,15), plot_over_image = False):
    """creates a plot showing the presence of water over time in a given area
    :param sar_dataset: xarray Dataset of sar data, with wasard values added from wasard_classify
    :param size: tuple indicating the size of the output plot
    :param plot_over_image: boolean indicating whether the wasard values will be plot on top of a landsat_dataset image, preventing a new figure from being drawn
    :return: None
    """
    
    # create an array containing the percent of time slices in which each pixel is predicted to hold water
    sar_dataset_copy      = sar_dataset.copy(deep=True)
    valid_time_indices    = _find_nodatas(sar_dataset_copy)
    aggregate_predictions = np.zeros((sar_dataset_copy.latitude.size, sar_dataset_copy.longitude.size))
    divisor               = len(valid_time_indices)
    
    for x in valid_time_indices:
        aggregate_predictions += np.nan_to_num(sar_dataset_copy.wasard[x])
            
    # divide total by number of valid datases in order to get the percent of the time each pixel holds water
    aggregate_predictions                     = aggregate_predictions / divisor
    sar_dataset_copy['aggregate_predictions'] = (('latitude', 'longitude'), aggregate_predictions)

    start_color_index     = 1
    if not plot_over_image:
        # set up figure to plot over
        start_time_string = str(sar_dataset.time.values[min(valid_time_indices)])[0:10]
        end_time_string   = str(sar_dataset.time.values[max(valid_time_indices)])[0:10]
        fig               = plt.figure(figsize=size)
        start_color_index = 0
        fig.suptitle('Water Frequency, {} to {}'.format(start_time_string, end_time_string, fontsize=20))
    
    # construct a plot using different colors to correspond to different flooding percentages
    color_pct = {0: 'black', 1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'blue'}
    for x in range(start_color_index,6):
        cond  = np.logical_and((1.0 * x-1)/5 < sar_dataset_copy.aggregate_predictions, sar_dataset_copy.aggregate_predictions <= (1.0*x)/5)
        water = sar_dataset_copy.aggregate_predictions.where(cond)
        try: 
            water.plot.imshow(levels = 5, colors = color_pct[x], add_colorbar = 0, add_labels=0)
        # throws ValueError when no values are plotted 
        except ValueError: 
            pass
    # previous loop missed some values where pixels always contained water(why?), so they're replotted here
    permanent_water = sar_dataset_copy.aggregate_predictions.where(sar_dataset_copy.aggregate_predictions == 1)
    try: 
        permanent_water.plot.imshow(levels = 5, colors = 'blue', add_colorbar = 0, add_labels=0)
    except ValueError: 
        pass
    
    print("% of time containing water:\nblack: 0%\nred: 0-20%\norange: 20-40%\nyellow: 40-60%\ngreen: 60-80%\nblue: 80-100%")
    
#specific names for sar_dataset
def get_correlation(sar_wasard_dataset, landsat_dataset, sar_time_index, landsat_time_index):
    """returns the percent of pixels from the sar_dataset scene_index that have the same predicted water value as the landsat_dataset scene_index
    :param landsat_dataset: xarray Dataset containing landsat data, loaded from datacube. If none, program predicts water values from the most recently trained classifier
    :param sar_dataset: xarray Dataset of sar data, with wasard values added from wasard_classify
    :param landsat_time_index: int indicating which time index from the landat Dataset will be used
    :param sar_time_index: int indicating which time index from the SAR dataset will be used
    :return: Ratio of pixels with the same predicted water value between the two scene_indexs to the total number of pixels 
    """
    
    
#    assert 'wasard' in sar_dataset.variable_names.keys()
    assert 'wasard' in sar_wasard_dataset.data_vars, "sar_dataset must include ""wasard"" datavar"
    
    landsat_dataset_at_time       = landsat_dataset.isel(time=landsat_time_index)
    landsat_dataset_wofs          = get_wofs_values(landsat_dataset_at_time)
    wofs_with_adjusted_resolution = _fit_landsat_dataset_resolution(landsat_dataset_wofs, sar_wasard_dataset)

    sar_dataset_at_time           = sar_wasard_dataset.isel(time=sar_time_index)
    
    
    # subtract wasard arrays of one dataset from the other, resulting array has value 0 when the wasard values were the same, and 1 when they were different
    differences_array             = wofs_with_adjusted_resolution - sar_dataset_at_time.wasard
    total                         = differences_array.size
    
    # generate dict containing the number of false positives, false negatives, and correlating values between each acquisition
    unique, counts                = np.unique(differences_array, return_counts=True)
    difference_counts             = dict(zip(unique, counts))
    result                        = {'False Positives':0, 'False Negatives':0, 'Correlating':0}
    result['False Positives']     = difference_counts[-1] / total
    result['False Negatives']     = difference_counts[1] / total
    result['Correlating']         = difference_counts[0] / total
    
    return result
   


def _find_training_indices(sar_dataset, landsat_dataset):
    """returns the optimal landsat and sentinel scene to train a new classifier on
    :param sar_datset: xarray Dataset containing landsat data, loaded from datacube 
    :param landsat_dataset: xarray dataset containing landsat data, loaded from datacube
    """
    
    cloud_cover_percentages              =  _get_cloud_avg(landsat_dataset)
    
    # Lambdas for filtering
    is_clear_enough                      = lambda time_index: cloud_cover_percentages[time_index] < 20
    if hasattr(sar_dataset, 'vv'):
        filter_nodata_sar                = lambda time_index: abs(np.sum(sar_dataset.vv[time_index])) > 10000
    filter_nodata_landsat                = lambda time_index: np.sum(landsat_dataset.red[time_index]) > 0
    
    # indices for selecting from datasets   
    sar_time_indices                     = range(sar_dataset.dims['time'])
    if hasattr(sar_dataset, 'vv'):
        sar_time_indices                 = filter(filter_nodata_sar, sar_time_indices)
    landsat_dataset_time_indices         = range(landsat_dataset.dims['time'])
    
    # filtering datasets
    landsat_dataset_time_indices         = filter(is_clear_enough, landsat_dataset_time_indices)
    landsat_dataset_time_indices         = filter(filter_nodata_landsat, landsat_dataset_time_indices)
    
    distance_in_time                     = lambda time_pair: abs(sar_dataset.time.values[time_pair[0]] 
                                                                 - landsat_dataset.time.values[time_pair[1]])
    cloud_cover                          = lambda time_pair: (cloud_cover_percentages[time_pair[1]], distance_in_time(time_pair))
    
    time_combinations                    = itertools.product(sar_time_indices, landsat_dataset_time_indices)
    time_combinations_distance_sorted    = sorted(time_combinations, key = distance_in_time)[0:5]
    
    for time_pair in time_combinations_distance_sorted:
        if cloud_cover(time_pair)[0] < 5:
            sar_index, landsat_dataset_index = time_pair
            # one of the top 5 pairs in terms of distance is below 5% cloud, no need to sort by cloud cover, so return indices now
           
            
            return (sar_index, landsat_dataset_index)
        
    time_combinations_cloud_cover_sorted = sorted(time_combinations_distance_sorted, key = cloud_cover)
    
    sar_index, landsat_dataset_index     = time_combinations_cloud_cover_sorted[0]
    
    return (sar_index, landsat_dataset_index)

def _get_cloud_avg(dataset):
    """generates a dict of cloud cover percentages over a given landsat dataset along with the average cloud cover over time
    :param dataset: xarray Dataset containing landsat data, loaded from datacube
    :return: dict of cloud cover percentage at each time slice
    """
    tot          = dataset.red[0].size
    time_indices = range(dataset.time.size)
    
    # use cf mask or pixel qa values to determine which pixels contain clouds
    if hasattr(dataset, 'cf_mask'):
        cloudy_pixels      = np.logical_or(dataset.cf_mask == 3, dataset.cf_mask == 4)
    else:
        acceptable_values = [322, 386, 834, 898, 1346 , 324, 388, 836, 900, 1348]
        avgs = {}
        global_mask = acceptable_values[0] == dataset.pixel_qa.values
        for bitmask in acceptable_values:
            global_mask = np.logical_or(global_mask, bitmask == dataset.pixel_qa.values)

        for time in range(global_mask.shape[0]):
            avg = np.count_nonzero(global_mask[time]) / global_mask[time].size
            avgs[time] = (1-avg) * 100

        return avgs   
    # determine cloud cover for each time slice
    cloud_pct  = lambda time_index: np.count_nonzero(cloudy_pixels[time_index]) / tot * 100
    cloud_avgs = {time_index: cloud_pct(time_index) for time_index in time_indices}
    return (cloud_avgs)

def get_clean_mask(landsat_dataset):
    acceptable_values = [322, 386, 834, 898, 1346 , 324, 388, 836, 900, 1348]
    global_mask = acceptable_values[0] == landsat_dataset.pixel_qa.values
    for bitmask in acceptable_values:
        global_mask = np.logical_or(global_mask, bitmask == landsat_dataset.pixel_qa.values)
    return global_mask

def get_wofs_values(landsat_dataset):
    """classifies a landsat scene using the wofs algorithm
    :param landsat_dataset: xarray with dims 'latitude','longitude' containing data from a landsat scene
    :return: xarray dataset containing wofs classification values
    """
    # landsat dataset needs dim 'time' for wofs_classify to work, re-add it here since using isel took it away
    landsat_dataset      = landsat_dataset.expand_dims('time')
    clean_mask           = None if hasattr(landsat_dataset, 'cf_mask') else get_clean_mask(landsat_dataset)
    landsat_dataset_wofs = wofs_classify(landsat_dataset, clean_mask = clean_mask)
    landsat_dataset_wofs = landsat_dataset_wofs.isel(time=0)
    return landsat_dataset_wofs



def _fit_landsat_dataset_resolution(landsat_dataset_wofs, sar_dataset):
    """adjusts the resolution of the landsat_dataset scene to fit that of the SAR scene
    :param landsat_dataset: xarray Dataset containing landsat data, loaded from datacube. If none, program predicts water values from the most recently trained classifier
    :param sar_dataset: xarray Dataset of sar data, with WASARD values added 
    :return: xarray dataset of the input landsat_dataset scene, stretched to match the resolution of the SAR scene
    """
   
    satellite_type                     = 'sentinel' if hasattr(sar_dataset, 'vv') else 'alos'
    wofs_values                        = landsat_dataset_wofs.wofs.values
    
    sar_dataset_rows                   = sar_dataset.latitude.size
    sar_dataset_columns                = sar_dataset.longitude.size
    landsat_dataset_rows               = landsat_dataset_wofs.latitude.size
    landsat_dataset_columns            = landsat_dataset_wofs.longitude.size
    resolution_multiplier              = round(max(sar_dataset_rows, landsat_dataset_rows) / 
                                               min(sar_dataset_rows, landsat_dataset_rows))
    
    wofs_values_columns_repeated       = np.repeat(wofs_values, resolution_multiplier, axis = 1)
    wofs_values_rows_columns_repeated  = np.repeat(wofs_values_columns_repeated, resolution_multiplier, axis = 0)
    wofs_values                        = wofs_values_rows_columns_repeated
    wofs_rows                          = wofs_values.shape[0]
    wofs_columns                       = wofs_values.shape[1]
   
    # truncate or lengthen landsat_dataset to fit sar_dataset values
    while wofs_rows    > sar_dataset_rows:
        random_row     = np.random.randint(0,wofs_rows)
        wofs_values    = np.delete(wofs_values, random_row, 0)
        wofs_rows      = wofs_values.shape[0]
        
    while wofs_columns > sar_dataset_columns:
        random_column  = np.random.randint(0,wofs_columns)
        wofs_values    = np.delete(wofs_values, random_column, 1)
        wofs_columns   = wofs_values.shape[1]

    while wofs_rows    < sar_dataset_rows:
        random_row     = np.random.randint(0,wofs_rows)
        wofs_values    = np.insert(wofs_values, random_row+1, wofs_values[random_row], axis=0)
        wofs_rows      = wofs_values.shape[0]
        
    while wofs_columns < sar_dataset_columns:
        random_column  = np.random.randint(0,wofs_columns)
        wofs_values    = np.insert(wofs_values, random_column+1, wofs_values[:,random_column], axis=1)
        wofs_columns   = wofs_values.shape[1]
    
    wofs_values_resolution_corrected = wofs_values
    
    return wofs_values_resolution_corrected



def _get_train_data(sar_dataset, landsat_dataset, pct = .09, displaydates = False, bands = 2, sar_time_index=-1, landsat_time_index=-1):
    """returns a tuple containing a training set and testing set to train a machine learning algorithm to detect water from SAR data
    :param sar_dataset: xarray of data from either a sar or ALOS satellite, loaded from the datacube
    :param landsat_dataset: xarray of landsat data, loaded from the datacube
    :param pct: float indicating the percentage of the dataset that should be used to train the machine learning algorithm. A higher percentage may increase accuracy as well as runtime
    :param bands: int specifying whether only the cross-pol channel or both the like-pol and cross-pol channels should be used when constructing the classifier
    :return: tuple containing a training set and testing set
    """
    
    # find a landsat_dataset scene with the lowest cloud cover to use as the truth values, and the sar_dataset scene closest in time to it
    no_index_provided = -1
    satellite_type = 'sentinel' if hasattr(sar_dataset, 'vv') else 'alos'

    
    if sar_time_index == no_index_provided:
        sar_time_index, landsat_time_index = _find_training_indices(sar_dataset, landsat_dataset)
    if displaydates:
        print("landsat_dataset acquisition date: {}".format(landsat_dataset.time.values[landsat_time_index]))
        print("SAR acquisition date: {}".format(sar_dataset.time.values[sar_time_index]))
    
    sar_dataset_at_time          = sar_dataset.isel(time=sar_time_index)
    landsat_dataset_at_time      = landsat_dataset.isel(time=landsat_time_index)
    landsat_dataset_wofs         = get_wofs_values(landsat_dataset_at_time)
    wofs_resolution_adjusted     = _fit_landsat_dataset_resolution(landsat_dataset_wofs, sar_dataset_at_time)
    truth_values                 = wofs_resolution_adjusted.flatten()
    
    band2 = None
    if satellite_type == 'sentinel':
        band1 = sar_dataset_at_time.vh.values.flatten()
        if bands == 2:
            band2 = sar_dataset_at_time.vv.values.flatten()
    if satellite_type == 'alos':
        band1 = sar_dataset_at_time.hv.values.flatten()
        # band2 = sar_dataset_at_time.incidence_angle.values.flatten()

    dset      = zip(band1, truth_values) if band2 is None else zip(band1, band2, truth_values)
    dset      = list(dset)
    
    has_water = lambda datapoint: datapoint[-1] == 1
    no_water  = lambda datapoint: datapoint[-1] == 0
    waters    = [datapoint for datapoint in dset if has_water(datapoint)]
    nonwaters = [datapoint for datapoint in dset if no_water(datapoint)]

    # sample nonwaters to be of equal length with waters in order to have an equal number of water and non water values to train with
    try:
        nonwaters_sampled = random.sample(nonwaters, len(waters))
    # Throws ValueError if there are more water pixels than nonwater pixels
    except ValueError:
        nonwaters_sampled = nonwaters
        waters            = random.sample(waters, len(nonwaters_sampled))
    
    # create a random sample of the values from the list
    train_pct      = .9
    test_pct       = 1 - train_pct
    full_dset      = waters + nonwaters_sampled
    shortened_dset = random.sample(full_dset, round(pct * len(full_dset)))
    train_data     = shortened_dset[0:round(train_pct*len(shortened_dset))]
    test_data      = shortened_dset[round(test_pct * len(shortened_dset)):len(shortened_dset)]
    
    if displaydates:
        print('training on {} samples'.format(len(train_data)))
    return (np.asarray(train_data), np.asarray(test_data))



def _filter_isolated_cells(array, struct, max_size):
    """ Return array with completely isolated blocks of cells removed
    :param array: Array with completely isolated single cells
    :param struct: Structure array for generating unique regions
    :param max_size: Int indicating how how small isolated blocks must be to be masked
    :return: Array with minimum region size > max_size
    """
    
    filtered_array                        = np.copy(array)
    id_regions, num_ids                   = scipy.ndimage.label(filtered_array, structure=struct)
    id_sizes                              = np.array(scipy.ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask                             = (id_sizes <= max_size)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def _filter_all(sar_dataset, max_size=50):
    """Filters max_sizelated blocks of predicted water values from a sar_dataset object's wasard array to try to remove false positives
    :param sar_dataset: xarray Dataset of sar data, with wasard values added from wasard_classify
    :param max_size: indicates maximum size of max_sizelated blocks to be filtered
    :return: new sar_dataset object with wasard values filtered for max_sizelated blocks
    """
    # sar_dataset_copy           = sar_dataset.copy(deep=True)
    dims                       = ('time','latitude','longitude')
    struct                     = np.ones((3,3))
    for x in range(sar_dataset.wasard.shape[0]):
                   sar_dataset.wasard[x] = _filter_isolated_cells(sar_dataset.wasard[x], struct, max_size)
    # sar_dataset['wasard'] = (dims, _filter_isolated_cells(sar_dataset.wasard, struct, max_size))
    return sar_dataset
    
    
def _find_nodatas(sentinel):
    """returns a list of the time slices of a SAR dataset which do not have a significant amount of nodata values"""
    vv_values            = sentinel.vv.values
    time_indices         = sentinel.time.values.size
    size                 = vv_values.size
    acceptable_threshold = .05*size
    is_acceptable        = lambda time_index: size - np.count_nonzero(vv_values[time_index]) > acceptable_threshold
    acceptable_indices   = [time_index for time_index in range(time_indices) if is_acceptable(time_index) ]
    return acceptable_indices


def _get_scores(sar_wasard_dataset, landsat_dataset, landsat_time_index):
    """returns the accuracy metrics for a SAR classifier
    :param landsat_dataset: xarray Dataset containing landsat data, loaded from datacube. If none, program predicts water values from the most recently trained classifier
    :param sar_dataset: xarray Dataset of sar data, with wasard values added from wasard_classify
    :param landsat_time_index: int indicating which time index from the landat Dataset will be used
    :param sar_time_index: int indicating which time index from the SAR dataset will be used
    :return: Ratio of pixels with the same predicted water value between the two scene_indexs to the total number of pixels 
    """
    
    
#    assert 'wasard' in sar_dataset.variable_names.keys()
    assert 'wasard' in sar_wasard_dataset.data_vars, "sar_dataset must include ""wasard"" datavar"
    
    landsat_dataset_at_time       = landsat_dataset.isel(time=landsat_time_index)
    landsat_dataset_wofs          = get_wofs_values(landsat_dataset_at_time)
    wofs_with_adjusted_resolution = _fit_landsat_dataset_resolution(landsat_dataset_wofs, sar_wasard_dataset)

    # sar_dataset_at_time           = sar_wasard_dataset.isel(time=sar_time_index)
    truth = wofs_with_adjusted_resolution.flatten()
    pred  = sar_wasard_dataset.wasard.values.flatten()
    
    pred1 = [pred[x] for x in range(len(pred)) if truth[x] >= 0]
    truth1 = [truth[x] for x in range(len(truth)) if truth[x] >= 0]
    
    
    precision = precision_score(truth1, pred1)
    recall    = recall_score(truth1, pred1)
    f1        = f1_score(truth1, pred1)
    
    return (f1, precision, recall)
    



