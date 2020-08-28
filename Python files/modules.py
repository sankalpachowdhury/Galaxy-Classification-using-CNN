# all dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
import os, random, shutil
from keras_preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib
import zipfile
import Augmentor
from keras.utils.vis_utils import plot_model
import matplotlib.image as img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import models
