import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import os
from PIL import Image
from tqdm import tqdm
import torch as torch
import torch.optim as optim
import torch.nn as nn
import os
import torchvision
from torch.autograd import Variable as var
import logging as log
import gc
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
# CUDA for PyTorch
def load_imagenet(model_mode):
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    import json
    ## Constants
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', transform=transform)
    generator=torch.Generator().manual_seed(0)
    train_data,test_data = torch.utils.data.random_split(dataset, [90000, 10000],generator=generator)
    batch_size = 200
    train_loader = DataLoader(train_data, batch_size=1000, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=0)
    x_train = []
    y_train = []
    for x_, y_ in train_loader:
        for x in x_:
            x = x.transpose(0, 2)
            x_train.append(x.numpy())
        for y in y_:
            y_train.append(int(y.numpy()))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test, y_test = [], []
    for x_, y_ in test_loader:
        for x in x_:
            x = x.transpose(0, 2)
            x_test.append(x.numpy())
        for y in y_:
            y_test.append(int(y.numpy()))
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=200)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=200)
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (x_train[80000:], y_train[80000:]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])
    m_train = np.ones(y_train.shape[0])
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train,y_train),(x_test,y_test),member

def loads_imagenet(model_mode):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    import json
    ## Constants
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', transform=transform)
    generator=torch.Generator().manual_seed(0)
    train_data,test_data = torch.utils.data.random_split(dataset, [90000, 10000],generator=generator)
    batch_size = 200
    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=0)
    x_train = []
    y_train = []
    for x_, y_ in train_loader:
        for x in x_:
            x = x.transpose(0, 2)
            x_train.append(x.numpy())
        for y in y_:
            y_train.append(int(y.numpy()))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test, y_test = [], []
    for x_, y_ in test_loader:
        for x in x_:
            x = x.transpose(0, 2)
            x_test.append(x.numpy())
        for y in y_:
            y_test.append(int(y.numpy()))
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    m_train = np.ones(y_train.shape[0])
    m_test = np.zeros(y_test.shape[0])

    x_train = x_train[:10000]
    y_train = y_train[:10000]

    label_x = np.zeros((4000, 32, 32, 3))
    label_y_train = np.ones(4000)
    label_y=sorted(enumerate(y_train), key=lambda x:x[1])
    label = []
    count_classes = []
    y = list(y_train)
    for i in range(200):
        count_classes.append(y.count(i))
    for i in range(len(label_y)):
        label.append(label_y[i][0])
    j = 0
    i = 0
    classes = 0
    count = 0
    while i < len(label):
        if count < 20:
            label_x[j] = x_train[label[i]]
            label_y_train[j] = int(label_y[i][1])
            j += 1
            i += 1
            count += 1
        else:
            count = 0
            i = sum(count_classes[:classes + 1])
            classes += 1
            if classes ==200:
                break
    y_train = label_y_train
    label_t_x = np.zeros((4000, 32, 32, 3))
    label_t_y_test = np.ones(4000)
    label_t_y = sorted(enumerate(y_test), key=lambda x: x[1])
    label_t = []
    for i in range(len(label_t_y)):
        label_t.append(label_t_y[i][0])
    j = 0
    i = 0
    count = 0
    classes = 0
    count_classes_test = []
    yytest = list(y_test)

    for i in range(200):
        count_classes_test.append(yytest.count(i))
    i=0
    while j < 4000:
        if count < 20:
            label_t_x[j] = x_test[label_t[i]]
            label_t_y_test[j] = int(label_t_y[i][1])
            j += 1
            i += 1
            count += 1
        else:
            count = 0
            i =  sum(count_classes_test[:classes + 1])
            classes += 1
            if classes == 200:
                break
    x_train=label_x
    x_test = label_t_x
    y_test = label_t_y_test
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=200)
    m_train = np.ones(y_train.shape[0])
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=200)
    m_test = np.zeros(y_test.shape[0])
    member = np.r_[m_train, m_test]
    member = np.r_[m_train, m_test]
    return x_train,y_train,x_test,y_test,member

def load_CH_MNIST(model_mode):
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    # Initialize Data
    tf.random.set_seed(1)
    images, labels = tfds.load('colorectal_histology', split='train', batch_size=-1, as_supervised=True)

    x_train, x_test, y_train, y_test = train_test_split(images.numpy(), labels.numpy(), train_size=0.5,
                                                        random_state=1 if model_mode == 'TargetModel' else 3,
                                                        stratify=labels.numpy())

    x_train = tf.image.resize(x_train, (32, 32))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=8)
    m_train = np.ones(y_train.shape[0])
    x_test = tf.image.resize(x_test, (32, 32))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=8)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member

def loads_CH_MNIST(model_mode):
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    # Initialize Data
    images, labels = tfds.load('colorectal_histology', split='train', batch_size=-1, as_supervised=True)

    x_train, x_test, y_train, y_test = train_test_split(images.numpy(), labels.numpy(), train_size=0.5,
                                                        random_state=1 if model_mode == 'TargetModel' else 3,
                                                        stratify=labels.numpy())

    x_train = tf.image.resize(x_train, (64, 64))
    x_test = tf.image.resize(x_test, (64, 64))
    label_x = tf.constant(0, shape=(2400, 64, 64, 3))
    label_y_train = np.ones(2400)
    label_x = np.array(label_x)
    label_y=sorted(enumerate(y_train), key=lambda x:x[1])
    label = []
    count_classes = []
    y = list(y_train)
    for i in range(8):
        count_classes.append(y.count(i))
    for i in range(len(label_y)):
        label.append(label_y[i][0])
    j = 0
    i = 0
    classes = 0
    count = 0
    while i < len(label):
        if count < 300:
            label_x[j] = x_train[label[i]]
            label_y_train[j] = int(label_y[i][1])
            j += 1
            i += 1
            count += 1
        else:
            count = 0
            i = sum(count_classes[:classes + 1])
            classes += 1
            if count_classes ==8:
                break
    x_train = tf.convert_to_tensor(label_x)
    y_train = label_y_train
    label_t_x = tf.constant(0, shape=(2400, 64, 64, 3))
    label_t_x = np.array(label_t_x)
    label_t_y_test = np.ones(2400)
    label_t_y = sorted(enumerate(y_test), key=lambda x: x[1])
    label_t = []
    for i in range(len(label_t_y)):
        label_t.append(label_t_y[i][0])
    j = 0
    i = 0
    count = 0
    classes = 0
    count_classes_test = []
    yytest = list(y_train)
    for i in range(8):
        count_classes_test.append(yytest.count(i))
    while j < 2400:
        if count < 300:
            label_t_x[j] = x_test[label_t[i]]
            label_t_y_test[j] = int(label_t_y[i][1])
            j += 1
            i += 1
            count += 1
        else:
            count = 0
            i =  sum(count_classes_test[:classes + 1])
            classes += 1
            if count_classes == 8:
                break
    label_t_x = tf.convert_to_tensor(label_t_x)
    x_test = label_t_x
    y_test = label_t_y_test
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=8)
    m_train = np.ones(y_train.shape[0])
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=8)
    m_test = np.zeros(y_test.shape[0])
    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member

def load_MNIST(model_mode):
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    # Initialize Data
    images, labels = tfds.load('mnist', split='train', batch_size=-1, as_supervised=True)

    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (images[40000:50000], labels[40000:50000]), \
                                               (images[50000:60000], labels[50000:60000])
    x_train = tf.image.resize(x_train, (32, 32))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    m_train = np.ones(y_train.shape[0])
    x_test = tf.image.resize(x_test, (32, 32))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member

def load_CIFAR(model_mode):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')
    tf.random.set_seed(1)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member

def loads_CIFAR(model_mode):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) =(x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])

    label_x=tf.constant(0,shape=(6000,32,32,3))
    label_y_train=np.ones(6000)
    label_x=np.array(label_x)
    label_y=sorted(enumerate(y_train), key=lambda x:x[1])
    label=[]
    count_classes=[]
    y=list(y_train)
    for i in range(100):
        count_classes.append(y.count(i))
    for i in range(len(label_y)):
        label.append(label_y[i][0])
    j=0
    i=0
    classes=0
    count=0
    while i<len(label):
        if count<60:
            label_x[j]=x_train[label[i]]
            label_y_train[j]=int(label_y[i][1])
            j+=1
            i+=1
            count+=1
        else:
            count=0
            i=sum(count_classes[:classes+1])
            classes+=1
            if count_classes==100:
                break
    x_train=tf.convert_to_tensor(label_x)
    y_train=label_y_train

    label_t_x=tf.constant(0,shape=(6000,32,32,3))
    label_t_x=np.array(label_t_x)
    label_t_y_test=np.ones(6000)
    label_t_y=sorted(enumerate(y_test), key=lambda x:x[1])
    label_t=[]
    for i in range(len(label_t_y)):
        label_t.append(label_t_y[i][0])
    j=0
    i=0
    count=0
    classes=0
    while j<6000:
        if count<60:
            label_t_x[j]=x_test[label_t[i]]
            label_t_y_test[j]=int(label_t_y[i][1])
            j+=1
            i+=1
            count+=1
        else:
            count=0
            i=100*(classes+1)
            classes+=1
            if count_classes==100:
                break
    label_t_x=tf.convert_to_tensor(label_t_x)
    x_test=label_t_x
    y_test=label_t_y_test

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member

def load_CIFAR10(model_mode):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')
    tf.random.set_seed(1)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member

def loads_CIFAR10(model_mode):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) =(x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])

    label_x=tf.constant(0,shape=(9000,32,32,3))
    label_y_train=np.ones(9000)
    label_x=np.array(label_x)
    label_y=sorted(enumerate(y_train), key=lambda x:x[1])
    label=[]
    count_classes=[]
    y=list(y_train)
    for i in range(10):
        count_classes.append(y.count(i))
    for i in range(len(label_y)):
        label.append(label_y[i][0])
    j=0
    i=0
    classes=0
    count=0
    while i<len(label):
        if count<900:
            label_x[j]=x_train[label[i]]
            label_y_train[j]=int(label_y[i][1])
            j+=1
            i+=1
            count+=1
        else:
            count=0
            i=sum(count_classes[:classes+1])
            classes+=1
            if count_classes==10:
                break
    x_train=tf.convert_to_tensor(label_x)
    y_train=label_y_train

    label_t_x=tf.constant(0,shape=(9000,32,32,3))
    label_t_x=np.array(label_t_x)
    label_t_y_test=np.ones(9000)
    label_t_y=sorted(enumerate(y_test), key=lambda x:x[1])
    label_t=[]
    for i in range(len(label_t_y)):
        label_t.append(label_t_y[i][0])
    j=0
    i=0
    count=0
    classes=0
    while j<9000:
        if count<900:
            label_t_x[j]=x_test[label_t[i]]
            label_t_y_test[j]=int(label_t_y[i][1])
            j+=1
            i+=1
            count+=1
        else:
            count=0
            i=1000*(classes+1)
            classes+=1
            if count_classes==10:
                break
    label_t_x=tf.convert_to_tensor(label_t_x)
    x_test=label_t_x
    y_test=label_t_y_test

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member

def load_CUB(model_mode):
    """
    Loads CALTECH_BIRDS2011 (CUB_200) dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    x_train, y_train = tfds.load('caltech_birds2011', split='train' if model_mode == 'TargetModel' else 'test',
                                 batch_size=-1, as_supervised=True)

    x_test, y_test = tfds.load('caltech_birds2011', split='test' if model_mode == 'TargetModel' else 'train',
                               batch_size=-1, as_supervised=True)

    x_train = tf.image.resize(x_train, (150, 150))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=200)
    m_train = np.ones(y_train.shape[0])

    x_test = tf.image.resize(x_test, (150, 150))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=200)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_EYE_PACS(model_mode):
    """
    Loads EyePACs dataset and maps it to Target Model and Shadow Model.
    If you would like to use this dataset, you could refer to the preprocess method mentioned in Kaggle.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')
    mode = "target" if model_mode == "TargetModel" else "shadow"

    img_folder = "data/Eye_PACs/{}_images/".format(mode)

    label_df = pd.read_csv("data/Eye_PACs/{}_label.csv".format(mode), index_col=0)

    def set_data(img_path, label, desired_size=150):
        N = len(os.listdir(img_path))
        x_ = np.empty((N, 150, 150, 3), dtype=np.uint8)
        y_ = np.empty(N)
        for i, img_name in enumerate(tqdm(os.listdir(img_path))):
            x_[i, :, :, :] = Image.open(img_path + img_name).resize((desired_size,) * 2, resample=Image.LANCZOS)
            y_[i] = label.loc[img_name, 'level']
        y_ = tf.keras.utils.to_categorical(y_, num_classes=5)

        return x_, y_

    x_, y_ = set_data(img_folder, label_df)
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, train_size=0.5,
                                                        random_state=1 if model_mode == "TargetModel" else 3,
                                                        stratify=y_)
    m_train = np.ones(y_train.shape[0])
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_Purchase_50(model_mode):
    """
    Loads Purchase dataset and maps it to Target Model and Shadow Model.
    This data comes from privacytrustlab/ml_privacy_meter, a simplified version.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `label_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    # Initialize Data
    dataframe = pd.read_csv('data/purchase_50_{}.csv'.
                            format("target"if model_mode == 'TargetModel' else "shadow"))
    trainDF, testDF = train_test_split(dataframe, train_size=0.5, random_state=1,
                                       stratify=dataframe['label'].values)
    x_train = trainDF.iloc[:, range(600)].values
    y_train = tf.keras.utils.to_categorical([i-1 for i in trainDF.loc[:, 'label']])
    m_train = np.ones(y_train.shape[0])

    x_test = testDF.iloc[:, range(600)].values
    y_test = tf.keras.utils.to_categorical([i-1 for i in testDF.loc[:, 'label']])
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_CIFAR_Ratio(model_mode, ratio=1):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    (x_train, y_train), (x_test, y_test) = (x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    x_train, y_train = x_train[:int(20000/(ratio+1))], y_train[:int(20000/(ratio+1))]

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return (x_train, y_train), (x_test, y_test), member


def load_CIFAR_Class(model_mode, num_classes=100):
    """
    Loads CIFAR-100 dataset and selects the concrete number of classes from it according to superclasses.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: Recommend to use one of [20, 40 ,60, 80, 100] and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    data = tfds.load("cifar100", split=["train", "test"], batch_size=-1)

    # train_index = [i in np.arange(int(num_classes/5)) for i in data[0]["coarse_label"]]
    # x_train = tf.boolean_mask(data[0]['image'], train_index)
    # train_label = tf.boolean_mask(data[0]['label'], train_index)
    # label_sorted = np.sort(np.unique(train_label))
    # y_train = tf.keras.utils.to_categorical([np.argwhere(label_sorted==i)[0] for i in train_label])
    #
    # test_index = [i in np.arange(int(num_classes/5)) for i in data[1]["coarse_label"]]
    # x_test = tf.boolean_mask(data[1]['image'], test_index)
    # test_label = tf.boolean_mask(data[1]['label'], test_index)
    # y_test = tf.keras.utils.to_categorical([np.argwhere(label_sorted==i)[0] for i in test_label])

    train_index = [i in np.arange(num_classes) for i in data[0]["label"]]
    x_train = tf.boolean_mask(data[0]['image'], train_index)
    train_label = tf.boolean_mask(data[0]['label'], train_index)
    label_sorted = np.sort(np.unique(train_label))
    y_train = tf.keras.utils.to_categorical([np.argwhere(label_sorted == i)[0] for i in train_label])

    test_index = [i in np.arange(num_classes) for i in data[1]["label"]]
    x_test = tf.boolean_mask(data[1]['image'], test_index)
    test_label = tf.boolean_mask(data[1]['label'], test_index)
    y_test = tf.keras.utils.to_categorical([np.argwhere(label_sorted == i)[0] for i in test_label])

    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (x_train[400*num_classes:500*num_classes],
                                                y_train[400*num_classes:500*num_classes]), (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:100*num_classes], y_train[:100*num_classes]), \
                                               (x_train[100*num_classes:200*num_classes],
                                                y_train[100*num_classes:200*num_classes])
    m_train = np.ones(y_train.shape[0])
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]

    return (x_train, y_train), (x_test, y_test), member
