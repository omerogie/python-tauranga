def movingaverage(interval, window_size):
    import numpy
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

def find_nearest_value(array, value):
    '''
    finds nearest value
    :param array:
    :param value:
    :return:
    '''
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_index(array, value):
    '''
    finds nearest index to value in array
    :param array:
    :param value:
    :return:
    '''
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx