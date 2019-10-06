import numpy as np
import pickle
import cv2
from keras import models
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class pred():
    def __init__(self):
        self.default_image_size = tuple((256, 256))
        with open('model/cnn_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
            self.model._make_predict_function()
        with open('model/label.pkl', 'rb') as f:
            self.label = pickle.load(f)

    def model_predict(self, filepath):


        np_img = np.array(self.convert_image_to_array(filepath), dtype=np.float16) / 225.0

        pred=self.model.predict(np.expand_dims(np_img, axis=0))
        pred=self.decode_result(pred[0])
        return pred

    def convert_image_to_array(self,image_dir):
        try:
            image = cv2.imread(image_dir)
            if image is not None:
                image = cv2.resize(image, self.default_image_size)
                return img_to_array(image)
            else:
                return np.array([])
        except Exception as e:
            print("Error :"+e)
            return None

    def decode_result(self,l):
        return self.label[np.where(l==max(l))[0][0]]