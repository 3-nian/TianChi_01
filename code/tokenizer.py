import pandas as pd
from config import *
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


# 生成语料集alldata
def alldata_preprocess():
    rawdata = pd.read_csv(train_data_file, names=["sn", "fault_time", "msg", "label"])
    rawdata['words'] = rawdata['msg'].str.replace(r'|', ' ')
    if LOCAL_TEST:
        rawdata = rawdata.loc[:100,:]
    del rawdata['sn']
    del rawdata['fault_time']
    del rawdata['msg']
    del rawdata['label']

    # 初赛
    preliminary_test_a_data = pd.read_csv(preliminary_test_a_data_file, names=["sn", "fault_time", "msg"])
    if LOCAL_TEST:
        preliminary_test_a_data = preliminary_test_a_data.loc[:100, :]
    preliminary_test_a_data['words'] = preliminary_test_a_data['msg'].str.replace(r'|', ' ')
    del preliminary_test_a_data['sn']
    del preliminary_test_a_data['fault_time']
    del preliminary_test_a_data['msg']

    preliminary_test_b_data = pd.read_csv(preliminary_test_b_data_file, names=["sn", "fault_time", "msg"])
    if LOCAL_TEST:
        preliminary_test_b_data = preliminary_test_b_data.loc[:100, :]
    preliminary_test_b_data['words'] = preliminary_test_b_data['msg'].str.replace(r'|', ' ')
    del preliminary_test_b_data['sn']
    del preliminary_test_b_data['fault_time']
    del preliminary_test_b_data['msg']

    all_value = rawdata['words'].append(preliminary_test_a_data['words']).append(preliminary_test_b_data['words'])

    if SUBMIT:
        # 决赛
        final_test_a_data = pd.read_csv(final_test_a_data_file, names=["sn", "fault_time", "msg"])
        if LOCAL_TEST:
            final_test_a_data = final_test_a_data.loc[:100,:]
        final_test_a_data['words'] = final_test_a_data['msg'].str.replace(r'|', ' ')
        del final_test_a_data['sn']
        del final_test_a_data['fault_time']
        del final_test_a_data['msg']

        final_test_b_data = pd.read_csv(final_test_b_data_file, names=["sn", "fault_time", "msg"])
        if LOCAL_TEST:
            final_test_b_data = final_test_b_data.loc[:100,:]
        final_test_b_data['words'] = final_test_b_data['msg'].str.replace(r'|', ' ')
        del final_test_b_data['sn']
        del final_test_b_data['fault_time']
        del final_test_b_data['msg']

        all_value = rawdata['words'].append(final_test_a_data['words']).append(final_test_b_data['words'])

    all_value.to_csv(USER_DATA_PATH + 'alldata.csv', index=False)


print('***************alldata starting***************')
alldata_preprocess()
print('***************alldata end***************')

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
###
tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

# 加入一些特殊字符
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=Vocab_Size)
# 空格分词器
tokenizer.pre_tokenizer = Whitespace()

print('***************tokenizer starting***************')
# 保存语料库文件
tokenizer.train([USER_DATA_PATH + 'alldata.csv'], trainer)
tokenizer.mask_token = '[MASK]'
tokenizer.save(USER_DATA_PATH + "tokenizer-my-Whitespace.json")
print('***************tokenizer end***************')
