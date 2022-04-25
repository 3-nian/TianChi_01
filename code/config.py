import torch
import numpy as np
import pandas as pd

# data file path
DATA_PATH = '../data/'
FEATURE_PATH = '../feature/'
USER_DATA_PATH = '../user_data/'
PREDICT_PATH = '../prediction_result/'
TC_DATA_PATH = '../tcdata/'

train_data_file = FEATURE_PATH + 'names_train16.csv'
test_index_file = FEATURE_PATH + 'test_index.pkl'
train_index_file = FEATURE_PATH + 'train_index.pkl'
preliminary_test_a_data_file = FEATURE_PATH + 'preliminary_test16_a.csv'
preliminary_test_b_data_file = FEATURE_PATH + 'preliminary_test16_b.csv'
final_test_a_data_file = FEATURE_PATH + 'final_test16_a.csv'
final_test_b_data_file = FEATURE_PATH + 'final_test16_b.csv'

# model setting
dropout = 0.1
Max_len = 256
Num_epoch = [5]
BATCH_SIZE = 32
Vocab_Size = 3000
num_labels = 4
metric_name = "f1"
n_splits = 10
random_state = 2022

# data_preprocess.py setting
LIMIT = 16  # 先选了LIMIT长度，后去的重，msg<=16
DUPLICATES = False  # 是否去重
REGULAR = False  # 是否正则清洗
ISPRINT = True  # 是否输出文件
JOIN = True  # 是否用|连接meg内容
VC = False  # 是否加入venus\crashdump作为msg内容
SUBMIT = False          #是否是提交环节，若是，则会加入tcdata部分
MILLIONMSG = True       #是否用百万msg生成tokenID
LOCAL_TEST = False

# cuda setting
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")


# metrics
def compute_metrics(eval_pred):
    weights = [5 / 11, 4 / 11, 1 / 11, 1 / 11]
    macro_F1 = 0.0
    predictions, labels = eval_pred
    #     [1,2,1,0] batch*1
    predictions = np.argmax(predictions, axis=1)
    overall_df = pd.DataFrame()
    overall_df['label_gt'] = labels
    overall_df['label_pr'] = predictions
    for i in range(len(weights)):
        TP = len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] == i)])
        FP = len(overall_df[(overall_df['label_gt'] != i) & (overall_df['label_pr'] == i)])
        FN = len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] != i)])
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        macro_F1 += weights[i] * F1
    return {'f1': macro_F1}
