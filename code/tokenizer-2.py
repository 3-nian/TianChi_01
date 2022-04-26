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
    # train数据集
    rawdata = pd.read_csv(DATA_PATH + 'preliminary_train/preliminary_sel_log_dataset.csv', usecols=['msg'])
    if LOCAL_TEST:
        rawdata = rawdata.loc[:100,:]
    rawdata['words'] = rawdata['msg'].str.replace(r'|', ' ')
    del rawdata['msg']

    # addition数据集
    adddata = pd.read_csv(DATA_PATH + 'preliminary_train/additional_sel_log_dataset.csv', usecols=['msg'])
    if LOCAL_TEST:
        adddata = adddata.loc[:100,:]
    adddata['words'] = adddata['msg'].str.replace(r'|', ' ')
    del adddata['msg']

    # 初赛数据集
    preliminary_test_a_data = pd.read_csv(DATA_PATH + 'preliminary_train/preliminary_sel_log_dataset_a.csv',usecols=['msg'])
    if LOCAL_TEST:
        preliminary_test_a_data = preliminary_test_a_data.loc[:100, :]
    preliminary_test_a_data['words'] = preliminary_test_a_data['msg'].str.replace(r'|', ' ')
    del preliminary_test_a_data['msg']

    preliminary_test_b_data = pd.read_csv(DATA_PATH + 'preliminary_train/preliminary_sel_log_dataset_b.csv',
                                          usecols=['msg'])
    if LOCAL_TEST:
        preliminary_test_b_data = preliminary_test_b_data.loc[:100, :]
    preliminary_test_b_data['words'] = preliminary_test_b_data['msg'].str.replace(r'|', ' ')
    del preliminary_test_b_data['msg']

    all_value = rawdata['words'].append(adddata['words']).append(preliminary_test_a_data['words']).append(
        preliminary_test_b_data['words'])
    if SUBMIT:
        # 决赛数据集
        final_test_a_data = pd.read_csv(TC_DATA_PATH + 'final_sel_log_dataset_a.csv', usecols=['msg'])
        if LOCAL_TEST:
            final_test_a_data = final_test_a_data.loc[:100,:]
        final_test_a_data['words'] = final_test_a_data['msg'].str.replace(r'|', ' ')
        del final_test_a_data['msg']

        final_test_b_data = pd.read_csv(TC_DATA_PATH + 'final_sel_log_dataset_b.csv', usecols=['msg'])
        if LOCAL_TEST:
            final_test_b_data = final_test_b_data.loc[:100,:]
        final_test_b_data['words'] = final_test_b_data['msg'].str.replace(r'|', ' ')
        del final_test_b_data['msg']

        all_value = all_value['words'].append(final_test_a_data['words']).append(final_test_b_data['words'])

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