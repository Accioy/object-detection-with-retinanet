import argparse
import os
import sys

import keras
import tensorflow as tf

#if __name__ == "__main__" and __package__ is None:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import keras_retinanet.bin  # noqa: F401
__package__ = "keras_retinanet.bin"

from .. import models
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import evaluate
from ..utils.keras_version import check_keras_version

# import keras_retinanet
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = models.load_model('resnet50_pascal_02_best_model_no_scale_test.h5', backbone_name='resnet50')
# keras.utils.plot_model(model,to_file='model.png',show_shapes=True)
# for i in range(0,len(model.layers)):
#     print(model.layers[i])
# imagefile="000682.jpg"
imagefile='000543.jpg'
IMAGE_DIR=os.path.join("F:\\yan\\keras-retinanet-master\\VOC2007\\JPEGImages",imagefile)
image = read_image_bgr(os.path.join(IMAGE_DIR))
image = preprocess_image(image)
image, scale = resize_image(image,min_side=500, max_side=600)

# boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
# boxes /= scale

# print(image.shape)
# (600, 497, 3)

image=np.expand_dims(image,axis=0)

from keras import backend as K

# P3到P7都可以这样搞，原代码见models.retinanet.py的line140
# C3到C5感觉比较麻烦，因为是直接掉keras的resnet模型，提取它的outputs实现的，具体调用在models.resnet.py的line99
layers=['P3','P4','P5','P6','P7']

for layer in layers:

    L=model.get_layer(name=layer)

    cla_submodel=model.get_layer(name='classification_submodel')
    # print(cla_submodel.summary())
    # for layer in cla_submodel.layers:
    #     print(layer.name)
    layer1 = cla_submodel.get_layer(name='pyramid_classification_sigmoid')
    feature = K.function([model.layers[0].input], [L.output])
    P=feature([image])[0]
    classi = K.function([cla_submodel.layers[0].input],[layer1.output])
    f1=classi([P])
    f1=f1[0][0]
    count=0
    for x in f1:
        if max(x)>0.5:
            count+=1
    print(layer,':',count)
    # f1=f1[0][0]
    # f1=np.transpose(f1,(2,0,1))
    # print(f1.shape)
    # for i in range(16):
    #     plt.subplot(4,4,i+1)
    #     plt.imshow(f1[i]) #,cmap='gray'  
    # plt.savefig(os.path.join("F:\\yan\\keras-retinanet-master\\visualization_results\\P_class_sig",layer+".jpg"))




