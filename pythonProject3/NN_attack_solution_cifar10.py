from dataLoader import *
from BlindMIUtil import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout, Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR10"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
SHADOW_MODEL_GENRE = sys.argv[3] if len(sys.argv) > 3 else "VGG16"
EPOCHS = 40
BATCH_SIZE = 64
NUM_CLASSES = 1
LEARNING_RATE = 5e-5
NN_ATTACK_WEIGHTS_PATH = "weights/NN_Attack/BlackBox/NN_Attack_{}_{}.hdf5".format(DATA_NAME, SHADOW_MODEL_GENRE)
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)
SHADOW_WEIGHTS_PATH = "weights/BlackShadow/{}_{}.hdf5".format(DATA_NAME, SHADOW_MODEL_GENRE)  #影子模型

(x_train_sha, y_train_sha), (x_test_sha, y_test_sha), m_train = globals()['load_' + DATA_NAME]('ShadowModel') #影子数据集
Shadow_Model = load_model(SHADOW_WEIGHTS_PATH)
c_train = np.sort(Shadow_Model.predict(np.r_[x_train_sha, x_test_sha]), axis=1)[:, ::-1]

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_test = globals()['load_' + DATA_NAME]('TargetModel') #训练集

x_train_tar=x_train_tar[:3000]
y_train_tar=y_train_tar[:3000]
labels=pd.read_excel('NN_incorrect_cifar10.xlsx')
labels=np.array(labels)[:,1]
labels=labels[:3000]
x_tt=[]
y_tt=[]
for i in labels:
    x_tt.append(x_test_tar[i])
    y_tt.append(y_test_tar[i])

x_test_tar=np.array(x_tt)
y_test_tar=np.array(y_tt)
m_test=m_test[:3000]
m_test=np.r_[m_test,np.zeros(3000)]

Target_Model = load_model(TARGET_WEIGHTS_PATH)
c_test = np.sort(Target_Model.predict(np.r_[x_train_tar, x_test_tar]), axis=1)[:, ::-1]

def create_attack_model(input_dim, num_classes=NUM_CLASSES):
    model = tf.keras.Sequential([
        Dense(512, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('sigmoid')
    ])
    model.summary()
    return model

def train(model, x_train, y_train):
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=[metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall()])
    checkpoint = ModelCheckpoint(NN_ATTACK_WEIGHTS_PATH, monitor='precision', verbose=1, save_best_only=True,
                                 mode='max')
    model.fit(x_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[checkpoint])


def evaluate(x_test, y_test):
    model = keras.models.load_model(NN_ATTACK_WEIGHTS_PATH)
    m_pred=model.predict(x_test)
    for i in range(len(m_pred)):
        if m_pred[i]>=0.5:
            m_pred[i]=1
        else:
            m_pred[i]=0
    y_pred=Target_Model.predict(np.r_[x_train_tar,x_test_tar])
    m,n=np.shape(y_pred)
    y_true=np.r_[y_train_tar,y_test_tar]
    '''
    y_pred=y_pred[10000:]
    y_true=y_true[10000:]
    index=[]
    for i in range(10000):
        interval = list(y_pred[i])
        index1 = interval.index(max(interval))
        true = list(y_true[i])
        true_index = true.index(max(true))
        if index1 != true_index:
            index.append(i)
    index=pd.DataFrame(index)
    index.to_excel('NN_incorrect.xlsx')
    '''
    count_correct=0
    count_error=0
    correct_member=0
    correct_nonmember=0
    incorrect_member=0
    incorrect_nonmember=0
    for i in range(m):
        interval=list(y_pred[i])
        index=interval.index(max(interval))
        true=list(y_true[i])
        true_index=true.index(max(true))
        if index==true_index:
            count_correct+=1
            if y_test[i]==1:
                correct_member+=1
            else:
                correct_nonmember+=1
        else:
            count_error+=1
            if y_test[i]==1:
                incorrect_member+=1
            else:
                incorrect_nonmember+=1
    nonmember_in_mem=0
    nonmember_correct_in_mem=0
    for i in range(m):
        if y_test[i]==0 and m_pred[i]==1:
            nonmember_in_mem+=1
            interval = list(y_pred[i])
            index = interval.index(max(interval))
            true = list(y_true[i])
            true_index = true.index(max(true))
            if index==true_index:
                nonmember_correct_in_mem+=1
    print('count_correct:',count_correct,'count_error:',count_error,'member:',sum(y_test),'nonmember:',m-sum(y_test),
          'correct_member:',correct_member,'incorrect_member:',incorrect_member,'correct_nonmember:',correct_nonmember,'incorrect_nonmember:',incorrect_nonmember,
          'precision:',count_correct/m,'nonmember_in_mem:',nonmember_in_mem,'nonmember_correct_in_mem:',nonmember_correct_in_mem,'total_member:',sum(m_pred),
          'error_ratio:',nonmember_in_mem/sum(m_pred),'nonmember_in_mem/nonmember:',nonmember_in_mem/(m-sum(y_test)))

    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=1)
    F1_Score = 2 * (precision * recall) / (precision + recall)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
          % (loss, accuracy, precision, recall, F1_Score))


attackModel = create_attack_model(c_train.shape[1])
train(attackModel, c_train, m_train)
evaluate(c_test, m_test)
