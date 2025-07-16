from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from BlindMIUtil import evaluate_attack
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
SHADOW_MODEL_GENRE = sys.argv[3] if len(sys.argv) > 2 else "VGG16"
NN_ATTACK_WEIGHTS_PATH = "weights/NN_Attack/NN_Attack_{}_{}.hdf5".format(DATA_NAME, SHADOW_MODEL_GENRE)
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)
SHADOW_WEIGHTS_PATH = "weights/BlackShadow/{}_{}.hdf5".format(DATA_NAME, SHADOW_MODEL_GENRE)

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_true = globals()['loads_' + DATA_NAME]('TargetModel')
Target_Model = load_model(TARGET_WEIGHTS_PATH)

x_=tf.constant(0,shape=(12000,32,32,3))
x_=np.array(x_)
y_=np.ones((12000,100))
m_=np.ones(12000)
i=0
j=0
while i<11999:
    x_[i]=x_train_tar[j]
    x_[i+1]=x_test_tar[j]
    m_[i+1]=0
    y_[i]=y_train_tar[j]
    y_[i+1]=y_test_tar[j]
    i+=2
    j+=1
x_=tf.convert_to_tensor(x_)
m_true=m_

def loss_threshold_attack(x_, y_true):
    (x_train_sha, y_train_sha), _, m_train = globals()['load_' + DATA_NAME]('ShadowModel')
    Shadow_Model = load_model(SHADOW_WEIGHTS_PATH)
    avg_loss = Shadow_Model.evaluate(x_train_sha, y_train_sha)[0]

    x_loss = np.asarray([-math.log(y_pred) if y_pred > 0 else y_pred+1e-50 for y_pred in Target_Model.
                        predict(x_)[y_true.astype(bool)]])
    m_pred = np.where(x_loss <= avg_loss, 1, 0)
    return m_pred

m_pred = loss_threshold_attack(x_, y_)

for i in range(len(m_pred)):
    if m_pred[i] >= 0.5:
        m_pred[i] = 1
    else:
        m_pred[i] = 0
y_pred = Target_Model.predict(x_)
m, n = np.shape(y_pred)
y_true = y_
count_correct = []
count_error = []
correct_member = []
correct_nonmember = []
incorrect_member = []
incorrect_nonmember = []
nonmember_in_mem = []
nonmember_correct_in_mem = []
member = []
nonmember = []
precision = []
total_member = []
error_ratio = []
nonmember_ratio = []
for k in range(100):
    index = 120 * k
    count_c = 0
    count_e = 0
    correct_m = 0
    correct_n = 0
    incorrect_m = 0
    incorrect_n = 0
    nonmember_in_m = 0
    nonmember_correct_in_m = 0
    member_ = sum(m_true[index:index + 120])
    nonmember_ = 120 - member_
    member.append(member_)
    nonmember.append(nonmember_)
    for i in range(index, index + 120):
        interval = list(y_pred[i])
        index = interval.index(max(interval))
        true = list(y_true[i])
        true_index = true.index(max(true))
        if index == true_index:
            count_c += 1
            if m_true[i] == 1:
                correct_m += 1
            else:
                correct_n += 1
        else:
            count_e += 1
            if m_true[i] == 1:
                incorrect_m += 1
            else:
                incorrect_n += 1

    for i in range(index, index + 120):
        if m_true[i] == 0 and m_pred[i] == 1:
            nonmember_in_m += 1
            interval = list(y_pred[i])
            index = interval.index(max(interval))
            true = list(y_true[i])
            true_index = true.index(max(true))
            if index == true_index:
                nonmember_correct_in_m += 1
    pre = count_c / 120
    count_correct.append(count_c)
    count_error.append(count_e)
    correct_member.append(correct_m)
    incorrect_member.append(incorrect_m)
    correct_nonmember.append(correct_n)
    incorrect_nonmember.append(incorrect_n)
    precision.append(pre)
    nonmember_in_mem.append(nonmember_in_m)
    nonmember_correct_in_mem.append(nonmember_correct_in_m)
    total_member.append(sum(m_pred[index:index + 120]))
    error_ratio.append(nonmember_in_m / sum(m_pred[index:index + 120]))
    nonmember_ratio.append(nonmember_in_m / (120 - sum(m_true[index:index + 120])))
result = np.c_[
    count_correct, count_error, member, nonmember, correct_member, incorrect_member, correct_nonmember, incorrect_nonmember, precision, nonmember_in_mem, nonmember_correct_in_mem, total_member, error_ratio, nonmember_ratio]
result = pd.DataFrame(result)
result.to_excel('Loss_Threshold_Attack_classes.xlsx')
evaluate_attack(m_pred, m_true)
