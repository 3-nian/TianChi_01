#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


LIMIT = 16
## 先选了LIMIT长度，后去的重，msg<=16
DUPLICATES = False
REGULAR = False
ISPRINT = True
print('data process starting ')
filepath = '../data/'

left = pd.read_csv(filepath + 'preliminary_train/preliminary_sel_log_dataset.csv',usecols=['sn','time','msg','server_model'])
right_1 = pd.read_csv(filepath + 'preliminary_train/preliminary_train_label_dataset.csv')
right_2 = pd.read_csv(filepath + 'preliminary_train/preliminary_train_label_dataset_s.csv')

right = pd.concat([right_1,right_2])

# print(left.count())
# print(right.count())
left.drop_duplicates(keep='last',inplace=True)
right.drop_duplicates(keep='last',inplace=True)
# print(left.count())
# print(right.count())
####################################################
testa_left = pd.read_csv('../tcdata/final_sel_log_dataset_a.csv',usecols=['sn','time','msg','server_model'])
testa_right = pd.read_csv('../tcdata/final_submit_dataset_a.csv')
# testb_left = pd.read_csv('final_sel_log_dataset_b.csv',usecols=['sn','time','msg','server_model'])
# testb_right = pd.read_csv('final_submit_dataset_b.csv')


# In[4]:


#查看类别分布
# plt.hist(right['label'], orientation = 'vertical', histtype = 'bar', color = 'red')
# plt.show()


# In[5]:


right['time']=right['fault_time']

##################################################
testa_right['time']=testa_right['fault_time']
# testb_right['time']=testb_right['fault_time']


# In[6]:


#全连接
data=pd.merge(left,right,on=['sn','time'],how='outer')
data.sort_values(by=['sn','time'],ascending=True,ignore_index=True,inplace=True)

#################################################3
testa_data=pd.merge(testa_left,testa_right,on=['sn','time'],how='outer')
testa_data.sort_values(by=['sn','time'],ascending=True,ignore_index=True,inplace=True)
# test_data.to_csv("./test_data_ftime_outer.csv",index=False,columns=['sn','time','msg','server_model','fault_time'],encoding="utf-8")
# testb_data=pd.merge(testb_left,testb_right,on=['sn','time'],how='outer')
# testb_data.sort_values(by=['sn','time'],ascending=True,ignore_index=True,inplace=True)


# In[7]:


#分组排序，fillna()  填充丢失/空值数据（可以选择填充个数，如10=故障类型只和前十个日志有关）,合并
# 优化

data['fault_time'] = data.groupby(data["sn"])['fault_time'].fillna(method='bfill',axis=0, limit=LIMIT)
# data['fault_time'] = data.groupby(data["sn"])['fault_time'].fillna(method='ffill',axis=0, limit=LIMIT)
#改回label试试
data['label'] = data.groupby(data["sn"])['label'].fillna(method='bfill',axis=0, limit=LIMIT)
# data['label'] = data.groupby(data["sn"])['label'].fillna(method='ffill',axis=0, limit=LIMIT)

# data.to_csv("./data_ftime_fb.csv",index=False,columns=['sn','time','msg','label','server_model','fault_time'],encoding="utf-8")
###############################################################test 没有 label
testa_data['fault_time'] = testa_data.groupby(testa_data["sn"])['fault_time'].fillna(method='bfill',axis=0, limit=LIMIT)
# test_data['fault_time'] = test_data.groupby(test_data["sn"])['fault_time'].fillna(method='ffill',axis=0, limit=LIMIT)
# test_data.to_csv("./test_data_ftime_fb.csv",index=False,columns=['sn','time','msg','server_model','fault_time'],encoding="utf-8")
# testb_data['fault_time'] = testb_data.groupby(testb_data["sn"])['fault_time'].fillna(method='bfill',axis=0, limit=LIMIT)


# In[8]:


data.drop(columns=['time','server_model'],axis=1,inplace=True)
# print(data.info())
###############################################
testa_data.drop(columns=['server_model'],axis=1,inplace=True)
# print(test_data.info())
# testb_data.drop(columns=['server_model'],axis=1,inplace=True)


# In[9]:


data.dropna(inplace=True)

#######################################################
testa_data.dropna(inplace=True)
# print(test_data.info())
# testb_data.dropna(inplace=True)


# In[10]:


if DUPLICATES:
    # 要不要去重？
    data.drop_duplicates(subset=['sn', 'fault_time','msg','label'], keep='first',inplace=True,ignore_index=True)
    # #####################################
    testa_data.drop_duplicates(subset=['sn', 'fault_time','msg'], keep='first',inplace=True,ignore_index=True)
    # testb_data.drop_duplicates(subset=['sn', 'fault_time','msg'], keep='first',inplace=True,ignore_index=True)


# In[11]:


if REGULAR:
    #删除数字括号，working off assert
    data['msg']=data['msg'].str.replace(';',' ').str.replace('[\(\)\d]+','').str.replace('\#\S* ','')    .str.replace('\@\S* ','').str.replace('\&\S* ','').str.replace('\-',' ').str.replace('_',' ').str.replace(',','')

    # feature_sec.drop_duplicates(keep='last',inplace=True,ignore_index=True)
    # print(data['msg']).str.replace('Front Panel','FP').str.replace('PCI\S*','PCI')
    ##################################################################
    testa_data['msg']=testa_data['msg'].str.replace(';',' ').str.replace('[\(\)\d]+','').str.replace('\#\S* ','')    .str.replace('\@\S* ','').str.replace('\&\S* ','').str.replace('\-',' ').str.replace('_',' ').str.replace(',','')
    # testb_data['msg']=testb_data['msg'].str.replace(';',' ').str.replace('[\(\)\d]+','').str.replace('\#\S* ','')    .str.replace('\@\S* ','').str.replace('\&\S* ','').str.replace('\-',' ').str.replace('_',' ').str.replace(',','')


# In[12]:



# 去|
data['msg'] = data['msg'].str.strip().str.replace(' \| ', '|')
data['msg'] = data['msg'].str.split('|')


##################################################################
testa_data['msg'] = testa_data['msg'].str.strip().str.replace(' \| ', '|')
testa_data['msg'] = testa_data['msg'].str.split('|')

# testb_data['msg'] = testb_data['msg'].str.strip().str.replace(' \| ', '|')
# testb_data['msg'] = testb_data['msg'].str.split('|')


# In[13]:


group_ftime = data.groupby([data["sn"],data["fault_time"],data["label"]])['msg'].apply(sum).reset_index()

# print(group_ftime.info())
# print(group_ftime.head())

##############################################################
testa_group_ftime = testa_data.groupby([testa_data["sn"],testa_data["fault_time"]])['msg'].apply(sum).reset_index()
# print(test_group_ftime.info())
# testb_group_ftime = testb_data.groupby([testb_data["sn"],testb_data["fault_time"]])['msg'].apply(sum).reset_index()


# In[14]:


# tmp = group_ftime['msg'].str.len()
# plt.hist(tmp, orientation = 'vertical', histtype = 'bar', color = 'red')
# plt.show()
# tmp.max()


# In[15]:


if REGULAR:
    #去碎片 
    group_ftime['msg_new'] = ''
    for i in range(group_ftime.shape[0]): 
        box = []
        for j in group_ftime['msg'][i]:
            if j != j.lower():
                box.append(j)
            else:
                if len(j) > 30:
                    box.append(j)
        group_ftime['msg_new'][i] = box 
    # print(group_ftime)
    group_ftime['msg'] = group_ftime['msg_new']
    group_ftime.drop(['msg_new'], axis=1, inplace=True)
    #########################################33

    # 去碎片 
    testa_group_ftime['msg_new'] = ''
    for i in range(testa_group_ftime.shape[0]): 
        testa_box = []
        for j in testa_group_ftime['msg'][i]:
            if j != j.lower():
                testa_box.append(j)
            else:
                if len(j) > 30:
                    testa_box.append(j)
        testa_group_ftime['msg_new'][i] = testa_box
    testa_group_ftime['msg'] = testa_group_ftime['msg_new']
    testa_group_ftime.drop(['msg_new'], axis=1,inplace=True)     
        # 去碎片 
    # testb_group_ftime['msg_new'] = ''
    # for i in range(testb_group_ftime.shape[0]):
    #     testb_box = []
    #     for j in testb_group_ftime['msg'][i]:
    #         if j != j.lower():
    #             testb_box.append(j)
    #         else:
    #             if len(j) > 30:
    #                 testb_box.append(j)
    #     testb_group_ftime['msg_new'][i] = testb_box
    # testb_group_ftime['msg'] = testb_group_ftime['msg_new']
    # testb_group_ftime.drop(['msg_new'], axis=1,inplace=True)


# In[16]:


for i in range(group_ftime.shape[0]): 
    group_ftime['msg'][i] = '|'.join(group_ftime['msg'][i])

###################################################3
for i in range(testa_group_ftime.shape[0]): 
    testa_group_ftime['msg'][i] = '|'.join(testa_group_ftime['msg'][i])
# for i in range(testb_group_ftime.shape[0]):
#     testb_group_ftime['msg'][i] = '|'.join(testb_group_ftime['msg'][i])


# In[17]:



sn_ftime_msg_label = group_ftime

# sn_ftime_msg_label.to_csv("./sn_ftime_msg_label_fb-raw.csv",index=False,columns=['sn','fault_time','msg','label'],encoding="utf-8")

####################################################3
testa_sn_ftime_msg_label = testa_group_ftime
# test_sn_ftime_msg_label.to_csv("./test_sn_ftime_msg_label_fb-raw.csv",index=False,columns=['sn','fault_time','msg'],encoding="utf-8")
# testb_sn_ftime_msg_label = testb_group_ftime


# In[18]:


sn_ftime_msg_label['label'] = sn_ftime_msg_label['label'].astype(int)
# sn_ftime_msg_label


# In[19]:


# testa_sn_ftime_msg_label


# In[20]:


# testb_sn_ftime_msg_label


# In[21]:
featurepath = '../feature/'

if ISPRINT:
    sn_ftime_msg_label.to_csv(featurepath + "names_train"+str(LIMIT)+".csv",
                               header=False, 
                               # compression='gzip',
                               index=False,
                               columns=['sn','fault_time','msg','label'],
                               encoding="utf-8")
    #############################################################################
    testa_sn_ftime_msg_label.to_csv(featurepath + "names_test"+str(LIMIT)+"_a.csv",
                               header=False, 
                               # compression='gzip',
                               index=False,
                               columns=['sn','fault_time','msg'],
                               encoding="utf-8"
                                  )

    # testb_sn_ftime_msg_label.to_csv("names_test"+str(LIMIT)+"_b.csv.gz",
    #                            header=False,
    #                            compression='gzip',
    #                            index=False,
    #                            columns=['sn','fault_time','msg'],
    #                            encoding="utf-8"
    #                               )
print('data process end ')
