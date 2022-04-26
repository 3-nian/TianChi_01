import pandas as pd
import numpy as np
from config import *

print('***************train data preprocess starting***************')

# 读取文件
venus = pd.read_csv(DATA_PATH + 'preliminary_train/preliminary_venus_dataset.csv',usecols=['sn','fault_time','module_cause','module'])
crashdump = pd.read_csv(DATA_PATH + 'preliminary_train/preliminary_crashdump_dataset.csv',usecols=['sn','fault_time','fault_code'])
venus['msg'] = venus['module_cause']
venus['time'] = venus['fault_time']
venus.drop(columns=['fault_time','module_cause','module'],axis=1,inplace=True)
crashdump['msg'] = crashdump['fault_code']
crashdump['time'] = crashdump['fault_time']
crashdump.drop(columns=['fault_time','fault_code'],axis=1,inplace=True)

left = pd.read_csv(DATA_PATH + 'preliminary_train/preliminary_sel_log_dataset.csv',usecols=['sn','time','msg'])
if VC:
    left = pd.concat([left,venus])
    left = pd.concat([left, crashdump])

right_1 = pd.read_csv(DATA_PATH + 'preliminary_train/preliminary_train_label_dataset.csv')
right_2 = pd.read_csv(DATA_PATH + 'preliminary_train/preliminary_train_label_dataset_s.csv')
right = pd.concat([right_1,right_2])
left.drop_duplicates(keep='last',inplace=True)
right.drop_duplicates(keep='last',inplace=True)


# 全连接需要time，label文件里只有fault_time
right['time']=right['fault_time']



#全连接
data=pd.merge(left,right,on=['sn','time'],how='outer')
data.sort_values(by=['sn','time'],ascending=True,ignore_index=True,inplace=True)



#分组排序，fillna()  填充丢失/空值数据（可以选择填充个数，如10=故障类型只和前十个日志有关）,合并
data['fault_time'] = data.groupby(data["sn"])['fault_time'].fillna(method='bfill',axis=0, limit=LIMIT)
data['label'] = data.groupby(data["sn"])['label'].fillna(method='bfill',axis=0, limit=LIMIT)

# 删除无用的列
data.drop(columns=['time'],axis=1,inplace=True)

# 去除空值
data.dropna(inplace=True)

if DUPLICATES:
    # 要不要去重？
    data.drop_duplicates(subset=['sn', 'fault_time','msg','label'], keep='first',inplace=True,ignore_index=True)

if REGULAR:
    #要不要正则表达式删除数字括号等等
    data['msg']=data['msg'].str.replace(';',' ').str.replace('[\(\)\d]+','').str.replace('\#\S* ','')    .str.replace('\@\S* ','').str.replace('\&\S* ','').str.replace('\-',' ').str.replace('_',' ').str.replace(',','')

# 删除|旁边的空格，按|分割成单词进行清洗
data['msg'] = data['msg'].str.strip().str.replace(' \| ', '|')
data['msg'] = data['msg'].str.split('|')

# 分组合并msg
group_ftime = data.groupby([data["sn"],data["fault_time"],data["label"]])['msg'].apply(sum).reset_index()


if REGULAR:
    #正则清理数据后删除残留碎片
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
    group_ftime['msg'] = group_ftime['msg_new']
    group_ftime.drop(['msg_new'], axis=1, inplace=True)

if JOIN:
    # 清洗完成后再用|连接
    for i in range(group_ftime.shape[0]):
        group_ftime['msg'][i] = '|'.join(group_ftime['msg'][i])

# 重命名准备输出
sn_ftime_msg_label = group_ftime
sn_ftime_msg_label['label'] = sn_ftime_msg_label['label'].astype(int)

# 输出为csv
if ISPRINT:
    sn_ftime_msg_label.to_csv(FEATURE_PATH + "names_train"+str(LIMIT)+".csv",
                               header=False,
                               index=False,
                               columns=['sn','fault_time','msg','label']
                              )
print('***************train data preprocess end***************')
