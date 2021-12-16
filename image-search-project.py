# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:11:32 2021

@author: mikson
"""

#libraries
import numpy as np
import os
import cv2 
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf #sql

from tqdm import tqdm_notebook #help to visualize train process or progress in the project
from scipy.spatial.distance import hamming, cosine #compare  vectors with hanging and crossing distance

#%matplotlib inline

def image_loader(image_path, image_size):
    '''
    Load an image from a disk.
    
    :param image_path: string - path to the image location
    :param image_size: tuple - size of an output image, (width,height) example: image_size=(32, 32)
    '''
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, image_size, cv2.INTER_CUBIC)
    return image
    
