import csv
import numpy as np
import cv2
import keras
import sys
import sklearn
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import load_model
from keras import optimizers
from keras import backend as K
from keras.models import model_from_json
K.set_image_dim_ordering('tf')

saved_model=sys.argv[1]
model=load_model(saved_model)
model.save_weights(saved_model+"_weights.h5")

model_json=model.to_json()

with open(saved_model+"_model.json","w") as json_file:
	json_file.write(model_json)
