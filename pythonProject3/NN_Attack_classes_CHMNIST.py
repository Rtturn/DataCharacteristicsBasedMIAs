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
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CH_MNIST"
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

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_test = globals()['loads_' + DATA_NAME]('TargetModel') #训练集
Target_Model = load_model(TARGET_WEIGHTS_PATH)
x_=tf.constant(0,shape=(4800,64,64,3))
x_=np.array(x_)
y_=np.ones((4800,8))
m_=np.ones(4800)
i=0
j=0
while i<4799:
    x_[i]=x_train_tar[j]
    x_[i+1]=x_test_tar[j]
    m_[i+1]=0
    y_[i]=y_train_tar[j]
    y_[i+1]=y_test_tar[j]
    i+=2
    j+=1
x_=tf.convert_to_tensor(x_)
m_true=m_
c_test = np.sort(Target_Model.predict(x_), axis=1)[:, ::-1]

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
    y_pred=Target_Model.predict(x_)
    m,n=np.shape(y_pred)
    y_true=y_
    count_correct=[]
    count_error=[]
    correct_member=[]
    correct_nonmember=[]
    incorrect_member=[]
    incorrect_nonmember=[]
    nonmember_in_mem=[]
    nonmember_correct_in_mem=[]
    member=[]
    nonmember=[]
    precision=[]
    total_member=[]
    error_ratio=[]
    nonmember_ratio=[]
    for k in range(8):
        index=600*k
        count_c= 0
        count_e= 0
        correct_m= 0
        correct_n= 0
        incorrect_m= 0
        incorrect_n= 0
        nonmember_in_m= 0
        nonmember_correct_in_m= 0
        member_=sum(y_test[index:index+600])
        nonmember_=600-member_
        member.append(member_)
        nonmember.append(nonmember_)
        for i in range(index,index+600):
            interval = list(y_pred[i])
            index = interval.index(max(interval))
            true = list(y_true[i])
            true_index = true.index(max(true))
            if index == true_index:
                count_c+= 1
                if y_test[i] == 1:
                    correct_m+= 1
                else:
                    correct_n+= 1
            else:
                count_e+= 1
                if y_test[i] == 1:
                    incorrect_m+= 1
                else:
                    incorrect_n+= 1
        for i in range(index,index+600):
            if y_test[i] == 0 and m_pred[i] == 1:
                nonmember_in_m+= 1
                interval = list(y_pred[i])
                index = interval.index(max(interval))
                true = list(y_true[i])
                true_index = true.index(max(true))
                if index == true_index:
                    nonmember_correct_in_m+= 1
        pre=count_c/600
        count_correct.append(count_c)
        count_error.append(count_e)
        correct_member.append(correct_m)
        incorrect_member.append(incorrect_m)
        correct_nonmember.append(correct_n)
        incorrect_nonmember.append(incorrect_n)
        precision.append(pre)
        nonmember_in_mem.append(nonmember_in_m)
        nonmember_correct_in_mem.append(nonmember_correct_in_m)
        total_member.append(sum(m_pred[index:index+600]))
        error_ratio.append(nonmember_in_m/sum(m_pred[index:index+600]))
        nonmember_ratio.append(nonmember_in_m/(600-sum(y_test[index:index+600])))
    result=np.c_[count_correct,count_error,member,nonmember,correct_member,incorrect_member,correct_nonmember,incorrect_nonmember,precision,nonmember_in_mem,nonmember_correct_in_mem,total_member,error_ratio,nonmember_ratio]
    result=pd.DataFrame(result)
    result.to_excel('NN_Attack_classes_chmnist.xlsx')
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=1)
    F1_Score = 2 * (precision * recall) / (precision + recall)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
          % (loss, accuracy, precision, recall, F1_Score))


attackModel = create_attack_model(c_train.shape[1])
train(attackModel, c_train, m_train)
evaluate(c_test, m_)
