import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.applications.resnet50 import preprocess_input
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



# EPOCHS = 25
# INIT_LR = 1e-3
# BS = 32
# default_image_size = tuple((256, 256))
# image_size = 0
# directory_root = '../input/plantvillage'
# width=256
# height=256
# depth=3
#
#
# image_list, label_list = [], []
# try:
#     print("[INFO] Loading images ...")
#     root_dir = listdir(directory_root)
#     for directory in root_dir:
#         # remove .DS_Store from list
#         if directory == ".DS_Store":
#             root_dir.remove(directory)
#
#     for disease_folder in root_dir:
#         # remove .DS_Store from list
#         if disease_folder == ".DS_Store":
#             plant_disease_folder_list.remove(disease_folder)
#
#     for plant_disease_folder in root_dir:
#         print(f"[INFO] Processing {plant_disease_folder} ...")
#         plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")
#         print(plant_disease_folder)
#         for single_plant_disease_image in plant_disease_image_list:
#             if single_plant_disease_image == ".DS_Store":
#                 plant_disease_image_list.remove(single_plant_disease_image)
#
#         for image in plant_disease_image_list[:200]:
#             print(image + "\n")
#             image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
#             if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
#                 image_list.append(convert_image_to_array(image_directory))
#                 label_list.append(plant_disease_folder)
#     print("[INFO] Image loading completed")
# except Exception as e:
#     print(f"Error : {e}")
#
#
#
# image_size = len(image_list)
#
# label_binarizer = LabelBinarizer()
# image_labels = label_binarizer.fit_transform(label_list)
# pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
# n_classes = len(label_binarizer.classes_)
#
# np_image_list = np.array(image_list, dtype=np.float16) / 225.0



# print("[INFO] Spliting data to train, test")
# x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42)
# x_test
#
#
#
# with open('cnn_model.pkl', 'rb') as f:
#     model = pickle.load(f)
#
#
# preds=model.predict(x_test)





class model:

    def __init__(self):

        #self.image
        with open('ref/model/cnn_model.pkl', 'rb') as f:
            self.model = pickle.load(f)


    #
    # def convert_image_to_array(self,image_dir):
    #     try:
    #         self.image = cv2.imread(image_dir)
    #         if self.image is not None :
    #             self.image = cv2.resize(self.image, default_image_size)
    #             return img_to_array(image)
    #         else :
    #             return np.array([])
    #     except Exception as e:
    #         print("Error : {e}")
    #     return None



    def model_predict(self,img_path ):
        print(img_path)
        self.img = image.load_img(img_path, target_size=(224, 224))

        np_img = image.img_to_array(self.img)
        batch_img = np.expand_dims(np_img, axis=0)
        processed_image = preprocess_input(batch_img.copy())
        preds = model.predict(processed_image)
        return preds
