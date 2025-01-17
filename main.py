# imports
import h5py
import numpy as np


# exploring hdf5 file
def checkfile(filename):
    with h5py.File(filename, 'r') as f:
        print(list(f.keys()))



# reading hdf5 file
checkfile('transfer cube human data/episode_0.hdf5')
