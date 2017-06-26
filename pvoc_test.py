from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import resnet as nn
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

all_imgs, classes_count, class_mapping = get_data(options.train_path)
