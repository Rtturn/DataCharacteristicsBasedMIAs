from dataLoader import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import metrics
from ModelUtil import *
import configparser
import sys
import torch as torch
import torch.optim as optim
import torch.nn as nn
import os
import torchvision
import torch.nn as nn
from torch.autograd import Variable as var
import logging as log
import gc
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
# CUDA for PyTorch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
config = configparser.ConfigParser()
config.read('target_model_config.ini')
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "imagenet"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
EPOCHS = int(config['{}_{}'.format(DATA_NAME, MODEL)]['EPOCHS'])
BATCH_SIZE = 200
LEARNING_RATE = float(config['{}_{}'.format(DATA_NAME, MODEL)]['LEARNING_RATE'])
WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, MODEL)


def train(model, x_train, y_train):
    """
    Train the target model and save the weight of the model
    :param model: the model that will be trained
    :param x_train: the image as numpy format
    :param y_train: the label for x_train
    :param weights_path: path to save the model file
    :return: None
    """
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=5e-5),
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
    model.fit(x_train,
              y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS)
    model.save(WEIGHTS_PATH)


def evaluate(x_test, y_test):
    model = keras.models.load_model(WEIGHTS_PATH)
    model.compile(loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=1)
    F1_Score = 2 * (precision * recall) / (precision + recall)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
          % (loss, accuracy, precision, recall, F1_Score))


x_train,y_train,x_test,y_test,member=globals()['load_' + DATA_NAME]()
TargetModel = globals()['create_{}_model'.format(MODEL)](x_train.shape[1:], y_train.shape[1])
train(TargetModel, x_train, y_train)
evaluate(x_train, y_train)
evaluate(x_test, y_test)