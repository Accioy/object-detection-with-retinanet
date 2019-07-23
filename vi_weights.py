import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import random
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

ROOT_DIR=os.path.abspath("./")
weights_file="resnet50_pascal_02_best_model_no_scale_test.h5"
model_path = os.path.join(ROOT_DIR,weights_file)

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
print(model.summary())

reg_submodel=model.get_layer(name='regression_submodel')
print(reg_submodel.summary())
layer1 = reg_submodel.get_layer(name='pyramid_regression_0')

cla_submodel=model.get_layer(name='classification_submodel')
print(cla_submodel.summary())
layer1 = cla_submodel.get_layer(name='pyramid_classification_0')

filters, bias=layer1.get_weights()

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
print(filters.shape)

number_of_image=1
for i in range(4):
    filter_i=filters[ : , : , : , i]
    for j in range(4):
        filter_i_j=filter_i[:,:,j]
        ax=plt.subplot(4,4,number_of_image)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(filter_i_j,cmap='gray')
        number_of_image+=1
plt.savefig(os.path.join("F:\\yan\\keras-retinanet-master\\visualization_results\\weights\\class_l1.jpg"))



