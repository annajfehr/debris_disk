import os
import numpy as np
from astropy.io import fits

class UVData:
    def __init__(self, directory):
        '''
        Initialize UVData object from the set of files in directory
        '''
        files = os.listdir(directory)
        self.n = len(files)

        self.datasets = []

        for i, f in enumerate(files):
            self.datasets.append(UVDataset(directory+f))

class UVDataset:
    def __init__(self, f):
        data_vis = fits.open(directory+f)
        self.header = data_vis.header
            
