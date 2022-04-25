import pandas as pd
import numpy as np


print('data process starting ')

LIMIT = 16                  #先选了LIMIT长度，后去的重，msg<=16
DUPLICATES = False          #是否去重
REGULAR = False             #是否正则清洗
ISPRINT = True              #是否输出文件
JOIN = True                 #是否用|连接meg内容
VC = False                  #是否加入venus\crashdump作为msg内容
filepath = '../data/'




# 读取文件
venus = pd.read_csv(filepath + 'preliminary_train/preliminary_venus_dataset.csv',usecols=['sn','fault_time','module_cause','module'])
crashdump = pd.read_csv(filepath + 'preliminary_train/preliminary_crashdump_dataset.csv',usecols=['sn','fault_time','fault_code'])
venus['msg'] = venus['module_cause']
venus['time'] = venus['fault_time']
venus.drop(columns=['fault_time','module_cause','module'],axis=1,inplace=True)
crashdump['msg'] = crashdump['fault_code']
crashdump['time'] = crashdump['fault_time']
crashdump.drop(columns=['fault_time','fault_code'],axis=1,inplace=True)

left = pd.read_csv(filepath + 'preliminary_train/preliminary_sel_log_dataset.csv',usecols=['sn','time','msg'])
if VC:
    left = pd.concat([left,venus])
    left = pd.concat([left, crashdump])

right_1 = pd.read_csv(filepath + 'preliminary_train/preliminary_train_label_dataset.csv')
right_2 = pd.read_csv(filepath + 'preliminary_train/preliminary_train_label_dataset_s.csv')
right = pd.concat([right_1,right_2])
left.drop_duplicates(keep='last',inplace=True)
right.drop_duplicates(keep='last',inplace=True)
####################################################
testa_left = pd.read_csv('../tcdata/final_sel_log_dataset_a.csv',usecols=['sn','time','msg'])
testa_right = pd.read_csv('../tcdata/final_submit_dataset_a.csv')
testb_left = pd.read_csv('../tcdata/final_sel_log_dataset_b.csv',usecols=['sn','time','msg'])
testb_right = pd.read_csv('../tcdata/final_submit_dataset_b.csv')

# 全连接需要time，label文件里只有fault_time
right['time']=right['fault_time']
##################################################
testa_right['time']=testa_right['fault_time']
testb_right['time']=testb_right['fault_time']


#全连接
data=pd.merge(left,right,on=['sn','time'],how='outer')
data.sort_values(by=['sn','time'],ascending=True,ignore_index=True,inplace=True)
#################################################3
testa_data=pd.merge(testa_left,testa_right,on=['sn','time'],how='outer')
testa_data.sort_values(by=['sn','time'],ascending=True,ignore_index=True,inplace=True)
testb_data=pd.merge(testb_left,testb_right,on=['sn','time'],how='outer')
testb_data.sort_values(by=['sn','time'],ascending=True,ignore_index=True,inplace=True)



#分组排序，fillna()  填充丢失/空值数据（可以选择填充个数，如10=故障类型只和前十个日志有关）,合并
data['fault_time'] = data.groupby(data["sn"])['fault_time'].fillna(method='bfill',axis=0, limit=LIMIT)
data['label'] = data.groupby(data["sn"])['label'].fillna(method='bfill',axis=0, limit=LIMIT)
###############################################################test 没有 label
testa_data['fault_time'] = testa_data.groupby(testa_data["sn"])['fault_time'].fillna(method='bfill',axis=0, limit=LIMIT)
testb_data['fault_time'] = testb_data.groupby(testb_data["sn"])['fault_time'].fillna(method='bfill',axis=0, limit=LIMIT)

# 删除无用的列
data.drop(columns=['time'],axis=1,inplace=True)
###############################################
# testa_data.drop(columns=['server_model'],axis=1,inplace=True)
# testb_data.drop(columns=['server_model'],axis=1,inplace=True)

# 去除空值
data.dropna(inplace=True)
#######################################################
testa_data.dropna(inplace=True)
testb_data.dropna(inplace=True)

if DUPLICATES:
    # 要不要去重？
    data.drop_duplicates(subset=['sn', 'fault_time','msg','label'], keep='first',inplace=True,ignore_index=True)
    # #####################################
    testa_data.drop_duplicates(subset=['sn', 'fault_time','msg'], keep='first',inplace=True,ignore_index=True)
    testb_data.drop_duplicates(subset=['sn', 'fault_time','msg'], keep='first',inplace=True,ignore_index=True)

if REGULAR:
    #要不要正则表达式删除数字括号等等
    data['msg']=data['msg'].str.replace(';',' ').str.replace('[\(\)\d]+','').str.replace('\#\S* ','')    .str.replace('\@\S* ','').str.replace('\&\S* ','').str.replace('\-',' ').str.replace('_',' ').str.replace(',','')
    ##################################################################
    testa_data['msg']=testa_data['msg'].str.replace(';',' ').str.replace('[\(\)\d]+','').str.replace('\#\S* ','')    .str.replace('\@\S* ','').str.replace('\&\S* ','').str.replace('\-',' ').str.replace('_',' ').str.replace(',','')
    testb_data['msg']=testb_data['msg'].str.replace(';',' ').str.replace('[\(\)\d]+','').str.replace('\#\S* ','')    .str.replace('\@\S* ','').str.replace('\&\S* ','').str.replace('\-',' ').str.replace('_',' ').str.replace(',','')


# 删除|旁边的空格，按|分割成单词进行清洗
data['msg'] = data['msg'].str.strip().str.replace(' \| ', '|')
data['msg'] = data['msg'].str.split('|')
##################################################################
testa_data['msg'] = testa_data['msg'].str.strip().str.replace(' \| ', '|')
testa_data['msg'] = testa_data['msg'].str.split('|')
testb_data['msg'] = testb_data['msg'].str.strip().str.replace(' \| ', '|')
testb_data['msg'] = testb_data['msg'].str.split('|')


# 分组合并msg
group_ftime = data.groupby([data["sn"],data["fault_time"],data["label"]])['msg'].apply(sum).reset_index()
##############################################################
testa_group_ftime = testa_data.groupby([testa_data["sn"],testa_data["fault_time"]])['msg'].apply(sum).reset_index()
testb_group_ftime = testb_data.groupby([testb_data["sn"],testb_data["fault_time"]])['msg'].apply(sum).reset_index()


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
    #########################################33
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
    for i in range(group_ftime.shape[0]):
        group_ftime['msg'][i] = '|'.join(group_ftime['msg'][i])
    ###################################################3
    for i in range(testa_group_ftime.shape[0]):
        testa_group_ftime['msg'][i] = '|'.join(testa_group_ftime['msg'][i])
    for i in range(testb_group_ftime.shape[0]):
        testb_group_ftime['msg'][i] = '|'.join(testb_group_ftime['msg'][i])



# 重命名准备输出
sn_ftime_msg_label = group_ftime
sn_ftime_msg_label['label'] = sn_ftime_msg_label['label'].astype(int)
####################################################3
testa_sn_ftime_msg_label = testa_group_ftime
testb_sn_ftime_msg_label = testb_group_ftime


# 输出为csv
featurepath = '../feature/'
if ISPRINT:
    sn_ftime_msg_label.to_csv(featurepath + "names_train"+str(LIMIT)+".csv",
                               header=False, 
                               index=False,
                               columns=['sn','fault_time','msg','label']
                              )
    #############################################################################
    testa_sn_ftime_msg_label.to_csv(featurepath + "names_test"+str(LIMIT)+"_a.csv",
                               header=False, 
                               index=False,
                               columns=['sn','fault_time','msg']
                                  )
    testb_sn_ftime_msg_label.to_csv("names_test"+str(LIMIT)+"_b.csv",
                               header=False,
                               index=False,
                               columns=['sn','fault_time','msg']
                                  )
print('data process end ')
