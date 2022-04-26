import pandas as pd
import numpy as np
from config import *

print('***************final data preprocess starting***************')

# 读取文件
testa_left = pd.read_csv(TC_DATA_PATH + 'final_sel_log_dataset_a.csv',usecols=['sn','time','msg'])
testa_right = pd.read_csv(TC_DATA_PATH + 'final_submit_dataset_a.csv')
testb_left = pd.read_csv(TC_DATA_PATH + 'final_sel_log_dataset_b.csv',usecols=['sn','time','msg'])
testb_right = pd.read_csv(TC_DATA_PATH + 'final_submit_dataset_b.csv')

# 全连接需要time，label文件里只有fault_time
testa_right['time']=testa_right['fault_time']
testb_right['time']=testb_right['fault_time']


#全连接
testa_data=pd.merge(testa_left,testa_right,on=['sn','time'],how='outer')
testa_data.sort_values(by=['sn','time'],ascending=True,ignore_index=True,inplace=True)
testb_data=pd.merge(testb_left,testb_right,on=['sn','time'],how='outer')
testb_data.sort_values(by=['sn','time'],ascending=True,ignore_index=True,inplace=True)



#分组排序，fillna()  填充丢失/空值数据（可以选择填充个数，如10=故障类型只和前十个日志有关）,合并
testa_data['fault_time'] = testa_data.groupby(testa_data["sn"])['fault_time'].fillna(method='bfill',axis=0, limit=LIMIT)
testb_data['fault_time'] = testb_data.groupby(testb_data["sn"])['fault_time'].fillna(method='bfill',axis=0, limit=LIMIT)

# 删除无用的列
# testa_data.drop(columns=['server_model'],axis=1,inplace=True)
# testb_data.drop(columns=['server_model'],axis=1,inplace=True)

# 去除空值
testa_data.dropna(inplace=True)
testb_data.dropna(inplace=True)

if DUPLICATES:
    # 要不要去重？
    testa_data.drop_duplicates(subset=['sn', 'fault_time','msg'], keep='first',inplace=True,ignore_index=True)
    testb_data.drop_duplicates(subset=['sn', 'fault_time','msg'], keep='first',inplace=True,ignore_index=True)

if REGULAR:
    #要不要正则表达式删除数字括号等等
    testa_data['msg']=testa_data['msg'].str.replace(';',' ').str.replace('[\(\)\d]+','').str.replace('\#\S* ','')    .str.replace('\@\S* ','').str.replace('\&\S* ','').str.replace('\-',' ').str.replace('_',' ').str.replace(',','')
    testb_data['msg']=testb_data['msg'].str.replace(';',' ').str.replace('[\(\)\d]+','').str.replace('\#\S* ','')    .str.replace('\@\S* ','').str.replace('\&\S* ','').str.replace('\-',' ').str.replace('_',' ').str.replace(',','')


# 删除|旁边的空格，按|分割成单词进行清洗
testa_data['msg'] = testa_data['msg'].str.strip().str.replace(' \| ', '|')
testa_data['msg'] = testa_data['msg'].str.split('|')
testb_data['msg'] = testb_data['msg'].str.strip().str.replace(' \| ', '|')
testb_data['msg'] = testb_data['msg'].str.split('|')

# 分组合并msg
testa_group_ftime = testa_data.groupby([testa_data["sn"],testa_data["fault_time"]])['msg'].apply(sum).reset_index()
testb_group_ftime = testb_data.groupby([testb_data["sn"],testb_data["fault_time"]])['msg'].apply(sum).reset_index()


if REGULAR:
    #正则清理数据后删除残留碎片
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

    testb_group_ftime['msg_new'] = ''
    for i in range(testb_group_ftime.shape[0]):
        testb_box = []
        for j in testb_group_ftime['msg'][i]:
            if j != j.lower():
                testb_box.append(j)
            else:
                if len(j) > 30:
                    testb_box.append(j)
        testb_group_ftime['msg_new'][i] = testb_box
    testb_group_ftime['msg'] = testb_group_ftime['msg_new']
    testb_group_ftime.drop(['msg_new'], axis=1,inplace=True)

if JOIN:
    # 清洗完成后再用|连接
    for i in range(testa_group_ftime.shape[0]):
        testa_group_ftime['msg'][i] = '|'.join(testa_group_ftime['msg'][i])
    for i in range(testb_group_ftime.shape[0]):
        testb_group_ftime['msg'][i] = '|'.join(testb_group_ftime['msg'][i])

# 重命名准备输出
testa_sn_ftime_msg_label = testa_group_ftime
testb_sn_ftime_msg_label = testb_group_ftime

# 输出为csv
if ISPRINT:
    testa_sn_ftime_msg_label.to_csv(FEATURE_PATH + "final_test"+str(LIMIT)+"_a.csv",
                               header=False,
                               index=False,
                               columns=['sn','fault_time','msg']
                                  )
    testb_sn_ftime_msg_label.to_csv(FEATURE_PATH + "final_test"+str(LIMIT)+"_b.csv",
                               header=False,
                               index=False,
                               columns=['sn','fault_time','msg']
                                  )
print('***************final data preprocess end***************')