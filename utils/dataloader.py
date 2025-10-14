import numpy as np
from scipy.io import loadmat

def load_ninapro_subject(filepath: str):
    mat = loadmat(filepath)
    emg = mat['emg']              
    labels = mat['restimulus'].ravel()  
    return emg, labels