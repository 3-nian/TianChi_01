#!/usr/bin/env python
# coding: utf-8

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



dropout = 0.1
data_file = '../feature/names_train16.csv'

test_index_file='../feature/test_index.pkl'
train_index_file='../feature/train_index.pkl'
final_test_data_file = '../feature/names_test16_a.csv'
Max_len = 256
Num_epoch = [5]
BATCH_SIZE = 32

gpu=0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")


# In[3]:


def data_preprocess():
    # rawdata = pd.read_csv(data_file, sep='\t', encoding='UTF-8',names=["sn", "fault_time", "msg", "SM"])
    rawdata = pd.read_csv(data_file,  encoding='UTF-8',names=["sn", "fault_time", "msg", "label"])
    #用正则表达式按标点替换文本
    import re
    # rawdata['words']=rawdata['text'].apply(lambda x: re.sub('3750|900|648',"",x))
    rawdata['words']=rawdata['msg'].str.replace(r'|',' ')#.to_string()
    del rawdata['sn']
    del rawdata['fault_time']
    del rawdata['msg']
    del rawdata['label']

    #预测
    final_test_data = pd.read_csv(final_test_data_file, encoding='UTF-8',names=["sn", "fault_time", "msg"])
    final_test_data['words'] = final_test_data['msg'].str.replace(r'|',' ')#.to_string()
    del final_test_data['sn']
    del final_test_data['fault_time']
    del final_test_data['msg']
    
    all_value= rawdata['words'].append(final_test_data['words'])
    all_value.to_csv('../user_data/alldata.csv',index=False)


# In[4]:
print('data_preprocess() starting ')

data_preprocess()
print('data_preprocess() end ')
from tokenizers import Tokenizer
from tokenizers.models import BPE,WordLevel
tokenizer= Tokenizer(BPE(unk_token="[UNK]"))
###
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

###
from tokenizers.trainers import BpeTrainer,WordLevelTrainer
#加入一些特殊字符
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],vocab_size=2048)

#空格分词器
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()
print('tokenizer starting ')
#保存语料库文件
tokenizer.train(['../user_data/alldata.csv'], trainer)
tokenizer.mask_token='[MASK]'
tokenizer.save("../user_data/tokenizer-my-Whitespace.json")
print('tokenizer end ')

# In[5]:


from transformers import PreTrainedTokenizerFast
#注意这里用了另外一种方式加载Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../user_data/tokenizer-my-Whitespace.json")
tokenizer.mask_token='[MASK]'
tokenizer.pad_token='[PAD]'
#数据预处理
from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train':'../user_data/alldata.csv'})


def preprocess_function(examples):
    return tokenizer(examples['words'], truncation=True,max_length=Max_len,padding='max_length')
encoded_dataset = dataset.map(preprocess_function, batched=True)
#数据收集器
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15 #mlm表示是否使用masked language model；mlm_probability表示mask的几率
)

#模型配置
# 自己修改部分配置参数
# config_kwargs = {
#     "cache_dir": None,
#     "revision": 'main',
#     "use_auth_token": None,
#     "hidden_size": 512,
#     "num_attention_heads": 2,
#     "hidden_dropout_prob": 0.2,
#     "vocab_size": tokenizer.get_vocab_size(), # 自己设置词汇大小
#     "embedding_size":64
# }
config_kwargs = {
    "d_model": 512,
    "n_head": 4,
    "vocab_size": 2048, #tokenizer.vocab_size, # 自己设置词汇大小
    "bi_data":False,
    "n_layer":8
}
# 将模型的配置参数载入
from transformers import RobertaConfig,AlbertConfig,XLNetConfig

config = XLNetConfig(**config_kwargs)
# 载入预训练模型，这里其实是根据某个模型结构调整config然后创建模型
from transformers import RobertaForMaskedLM,AlbertForMaskedLM,XLNetLMHeadModel

model = XLNetLMHeadModel(config=config)
# model = AutoModelForMaskedLM.from_pretrained(
#             'pretrained_bert_models/bert-base-chinese/',
#             from_tf=bool(".ckpt" in 'roberta-base'), # 支持tf的权重
#             config=config,
#             cache_dir=None,
#             revision='main',
#             use_auth_token=None,
#         )
# model.resize_token_embeddings(len(tokenizer))
#output:Embedding(863, 768, padding_idx=1)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="../user_data/BERT",
    overwrite_output_dir=True,
    num_train_epochs=5, #训练epoch次数
    per_device_train_batch_size=BATCH_SIZE, #训练时的batchsize
    save_steps=10_000, #每10000步保存一次模型
    save_total_limit=2,#最多保存两次模型
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator, #数据收集器在这里
    train_dataset=encoded_dataset["train"] #注意这里选择的是预处理后的数据集
)
print('trainer pretraining starting ')
#开始训练
# trainer.train()
# #保存模型
# trainer.save_model("../user_data/BERT")

print('trainer pretraining end ')
# In[4]:


def data_preprocess():
    rawdata = pd.read_csv(data_file,  encoding='UTF-8',names=["sn", "fault_time", "msg", "label"])

    #用正则表达式按标点替换文本
    import re
    rawdata['words']=rawdata['msg'].str.replace(r'|',' ')
    del rawdata['msg']
    del rawdata['sn']
    del rawdata['fault_time']
    rawdata['label'] = rawdata['label'].astype(int)
    #数据划分
    #如果之前已经做了就直接加载
#     if os.path.exists(test_index_file) and os.path.exists(train_index_file):
#         test_index=joblib.load(test_index_file)
#         train_index=joblib.load(train_index_file)
#     else:
    rawdata.reset_index(inplace=True, drop=True)
#     X = list(rawdata.index)
#     y = rawdata['label']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
#                                                         stratify=y)  # stratify=y表示分层抽样，根据不同类别的样本占比进行抽样
#     test_index = {'X_test': X_test, 'y_test': y_test}
#     joblib.dump(test_index, 'test_index.pkl')
#     train_index = {'X_train': X_train, 'y_train': y_train}
#     joblib.dump(train_index, 'train_index.pkl')

#     train_x=rawdata.loc[train_index['X_train']]
#     train_y=rawdata.loc[train_index['X_train']]['label'].values

#     X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1,
#                                                         stratify=train_y)
#     #训练集
#     X_train.columns=['label', 'words']
#     X_train.to_csv('train_data.csv',index=False)
#     #开发集
#     X_test.columns=['label', 'words']
#     X_test.to_csv('dev_data.csv',index=False)
#     #测试集
#     test_x=rawdata.loc[test_index['X_test']]
#     test_x.columns=['label', 'words']
#     test_x.to_csv('test_data.csv',index=False)
    #预测
    f = pd.read_csv(final_test_data_file,  encoding='UTF-8',names=["sn", "fault_time", "msg"])
    f['words'] = f['msg'].str.replace(r'|', ' ')
    del f['msg']
    del f['sn']
    del f['fault_time']
    f.to_csv('../user_data/final_test_data.csv',index=False)
    return rawdata


# In[5]:


# def compute_metrics(eval_pred):
# #     print('*'*50)
# #     print(eval_pred.shape)
#     metric = load_metric('f1.py')
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return metric.compute(predictions=predictions, references=labels,average='macro')

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


# In[6]:


#注意这里用了另外一种方式加载Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../user_data/tokenizer-my-Whitespace.json")
tokenizer.mask_token='[MASK]'
tokenizer.pad_token='[PAD]'

def preprocess_function(examples):
    return tokenizer(examples['words'], truncation=True,max_length=Max_len,padding='max_length')


model_checkpoint = "../user_data/BERT"#"BERT" #所选择的预训练模型

num_labels = 4
batch_size = 128
metric_name = "f1"
# model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

def get_arg(num,train=True):
    train_args = TrainingArguments(
        '../user_data/' + str(num) + "_test-glue",
        evaluation_strategy = "epoch", #每个epcoh会做一次验证评估；
        save_strategy = "epoch",
        logging_dir = 'test-glue/log',
        logging_strategy = "epoch",
        report_to="tensorboard",
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=num,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name, #根据哪个评价指标选最优模型
        save_steps=10_000,
        save_total_limit=2,
        dataloader_drop_last = True
    )

    test_args = TrainingArguments(
        '../user_data/' + str(num) + "_test-glue",
        evaluation_strategy = "epoch", #每个epcoh会做一次验证评估；
        save_strategy = "epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=num,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name, #根据哪个评价指标选最优模型
        save_steps=10_000,
        save_total_limit=2,
        no_cuda = True
    )
    return train_args if train else test_args

# 10折交叉验证
n_splits = 10
random_state = 2022
kf = StratifiedKFold(n_splits,shuffle=True,random_state=random_state)
raw_data = data_preprocess()
y = raw_data['label']
train_loss = []
test_f1 = []
best_list = []


X_train, X_test, y_train, y_test = train_test_split(raw_data, y, test_size=0.1,stratify=y)

#测试集
X_test.columns=['label', 'words']
X_test.to_csv('../user_data/test_data.csv',index=False)
X_train = pd.DataFrame(X_train)
for epoch in Num_epoch:
    best_f1 = 0.0
    for k, (train_index, test_index) in enumerate(kf.split(X_train,y_train)):
        train_data,test_data = X_train.iloc[train_index],X_train.iloc[test_index]

        #训练集
        train_data.columns=['label', 'words']
        train_data.to_csv('../user_data/train_data.csv',index=False)
        #开发集
        test_data.columns=['label', 'words']
        test_data.to_csv('../user_data/dev_data.csv',index=False)


        dataset = load_dataset('csv', data_files={'train':'../user_data/train_data.csv',
                                              'dev':'../user_data/dev_data.csv',
                                              'test':'../user_data/test_data.csv'}) #这里建议用完全路径，否则可能卡住
        encoded_dataset = dataset.map(preprocess_function, batched=True)


        model_checkpoint = "../user_data/BERT" #if k==0 else str(epoch) + "_test-glue"#"BERT" #所选择的预训练模型
        # model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

        trainer = Trainer(
            model,
            get_arg(epoch),
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["dev"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        # train_result = trainer.train()
        # loss_train = train_result.training_loss
        # train_loss.append(loss_train)

        trainer = Trainer(
            model,
            get_arg(epoch,train=False),
            eval_dataset=encoded_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        # test_result = trainer.evaluate()
        # score_test = test_result['eval_f1']
        # test_f1.append(score_test)
        # best_list.append(best_f1)
        # if score_test > best_f1:
        #     best_f1 = score_test
        #     trainer.save_model('../user_data/' + str(epoch) + "_test-glue")
        # print(k, " 折", " train loss:   ", loss_train)
        print(k, " 折", " best F1:   ", best_f1)
        # print(k, " 折", " test F1:   ", score_test, '\n')

