from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from BlindMIUtil import evaluate_attack
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_true = globals()['load_' + DATA_NAME]('TargetModel')
Target_Model = load_model(TARGET_WEIGHTS_PATH)

x_train_tar=x_train_tar[:7000]
y_train_tar=y_train_tar[:7000]
labels=pd.read_excel('NN_incorrect.xlsx')
labels=np.array(labels)[:,1]
labels=labels[:7000]
x_tt=[]
y_tt=[]
for i in labels:
    x_tt.append(x_test_tar[i])
    y_tt.append(y_test_tar[i])

x_test_tar=np.array(x_tt)
y_test_tar=np.array(y_tt)
m_true=m_true[:7000]
m_true=np.r_[m_true,np.zeros(7000)]

def top1_threshold_attack(x_, target):
    nonM_generated = np.random.uniform(0, 255, (1000, )+x_.shape[1:])
    threshold = np.percentile(target.predict(nonM_generated).max(axis=1), 90, interpolation='lower')
    m_pred = np.where(target.predict(x_).max(axis=1)>threshold, 1, 0)
    return m_pred


m_pred = top1_threshold_attack(np.r_[x_train_tar, x_test_tar], Target_Model)

for i in range(len(m_pred)):
    if m_pred[i] >= 0.5:
        m_pred[i] = 1
    else:
        m_pred[i] = 0
y_pred = Target_Model.predict(np.r_[x_train_tar, x_test_tar])
m, n = np.shape(y_pred)
y_true = np.r_[y_train_tar, y_test_tar]
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
count_correct = 0
count_error = 0
correct_member = 0
correct_nonmember = 0
incorrect_member = 0
incorrect_nonmember = 0
for i in range(m):
    interval = list(y_pred[i])
    index = interval.index(max(interval))
    true = list(y_true[i])
    true_index = true.index(max(true))
    if index == true_index:
        count_correct += 1
        if m_true[i] == 1:
            correct_member += 1
        else:
            correct_nonmember += 1
    else:
        count_error += 1
        if m_true[i] == 1:
            incorrect_member += 1
        else:
            incorrect_nonmember += 1
nonmember_in_mem = 0
nonmember_correct_in_mem = 0
for i in range(m):
    if m_true[i] == 0 and m_pred[i] == 1:
        nonmember_in_mem += 1
        interval = list(y_pred[i])
        index = interval.index(max(interval))
        true = list(y_true[i])
        true_index = true.index(max(true))
        if index == true_index:
            nonmember_correct_in_mem += 1
print('count_correct:', count_correct, 'count_error:', count_error, 'member:', sum(m_true), 'nonmember:',
      m - sum(m_true),
      'correct_member:', correct_member, 'incorrect_member:', incorrect_member, 'correct_nonmember:', correct_nonmember,
      'incorrect_nonmember:', incorrect_nonmember,
      'precision:', count_correct / m, 'nonmember_in_mem:', nonmember_in_mem, 'nonmember_correct_in_mem:',
      nonmember_correct_in_mem, 'total_member:', sum(m_pred),
      'error_ratio:', nonmember_in_mem / sum(m_pred), 'nonmember_in_mem/nonmember:',
      nonmember_in_mem / (m - sum(m_true)))
evaluate_attack(m_pred, m_true)