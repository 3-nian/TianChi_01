import torch
#加载训练数据
import joblib
import pandas as pd
import os
from datasets import load_dataset
# 将模型的配置参数载入
from transformers import RobertaConfig,AlbertConfig,XLNetConfig

# 载入预训练模型，这里其实是根据某个模型结构调整config然后创建模型
from transformers import RobertaForMaskedLM,AlbertForMaskedLM,XLNetLMHeadModel
from sklearn.model_selection import train_test_split


from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import XLNetForSequenceClassification, TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
from sklearn.model_selection import  StratifiedKFold
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dropout = 0.1
# data_file = '../feature/names_train16.csv'
# final_test_data_file = '../feature/names_test16_a.csv'
Max_len = 256


def compute_metrics(eval_pred):
    weights =  [5  /  11,  4  /  11,  1  /  11,  1  /  11]
    macro_F1 =  0.0
    predictions, labels = eval_pred
#     [1,2,1,0] batch*1
    predictions = np.argmax(predictions, axis=1)
    overall_df = pd.DataFrame()
    overall_df['label_gt'] = labels
    overall_df['label_pr'] = predictions
    for i in  range(len(weights)):
        TP =  len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] == i)])
        FP =  len(overall_df[(overall_df['label_gt'] != i) & (overall_df['label_pr'] == i)])
        FN =  len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] != i)])
        precision = TP /  (TP + FP)  if  (TP + FP)  >  0  else  0
        recall = TP /  (TP + FN)  if  (TP + FN)  >  0  else  0
        F1 =  2  * precision * recall /  (precision + recall)  if  (precision + recall)  >  0  else  0
        macro_F1 += weights[i]  * F1
    return {'f1':macro_F1}

#注意这里用了另外一种方式加载Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../user_data/tokenizer-my-Whitespace.json")
tokenizer.mask_token='[MASK]'
tokenizer.pad_token='[PAD]'

def preprocess_function(examples):
    return tokenizer(examples['words'], truncation=True,max_length=Max_len,padding='max_length')


model_checkpoint = "../user_data/20_test-glue"#"BERT" #所选择的预训练模型

num_labels = 4
batch_size = 16
metric_name = "f1"
model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

test_args = TrainingArguments(
    "../user_data/20_test-glue",
#     evaluation_strategy = "epoch", #每个epcoh会做一次验证评估；
#     save_strategy = "epoch",
#     learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
#     load_best_model_at_end=True,
    metric_for_best_model=metric_name, #根据哪个评价指标选最优模型
#     save_steps=10_000,
#     save_total_limit=2
#     no_cuda = True
)




dataset = load_dataset('csv', data_files={'train':'../user_data/train_data.csv',
                                              'dev':'../user_data/dev_data.csv',
                                              'test':'../user_data/test_data.csv'}) #这里建议用完全路径，否则可能卡住
encoded_dataset = dataset.map(preprocess_function, batched=True)

trainer = Trainer(
    model,
    test_args,
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(trainer.evaluate())

# print('************predict starting********************')
# final_test_dataset = load_dataset('csv', data_files={'res':'../user_data/final_test_data.csv'})#,cache_dir='res-fine-tune'
# encoded_final_test_dataset = final_test_dataset.map(preprocess_function, batched=True)
# res=trainer.predict(test_dataset=encoded_final_test_dataset["res"])
# csv=pd.DataFrame(np.argmax(res[0],1),columns=['label'])
# print('************predict end********************')
# test_data = pd.read_csv('../feature/names_test16_a.csv',header=None,names=['sn','fault_time','msg'])
# test_data['label'] = csv
# submit_data = pd.read_csv('../tcdata/final_submit_dataset_a.csv')
# submit_final_data = pd.merge(submit_data,test_data,on=['sn','fault_time'],how='left')
# submit_final_data.sort_values(by=['sn','fault_time'],inplace=True)
# submit_final_data['label'].fillna(method='ffill',axis=0,inplace=True)
# submit_final_data.drop(['msg'],axis=1,inplace=True)
# submit_final_data.to_csv('../prediction_result/predictions.csv',index=False)