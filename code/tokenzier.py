import pandas as pd




# 生成语料集alldata
def alldata_preprocess():
    rawdata = pd.read_csv(train_data_file, names=["sn", "fault_time", "msg", "label"])
    rawdata['words'] = rawdata['msg'].str.replace(r'|', ' ')
    del rawdata['sn']
    del rawdata['fault_time']
    del rawdata['msg']
    del rawdata['label']

    # 预测
    final_test_a_data = pd.read_csv(final_test_a_data_file, names=["sn", "fault_time", "msg"])
    final_test_a_data['words'] = final_test_a_data['msg'].str.replace(r'|', ' ')
    del final_test_a_data['sn']
    del final_test_a_data['fault_time']
    del final_test_a_data['msg']

    final_test_b_data = pd.read_csv(final_test_b_data_file, names=["sn", "fault_time", "msg"])
    final_test_b_data['words'] = final_test_b_data['msg'].str.replace(r'|', ' ')
    del final_test_b_data['sn']
    del final_test_b_data['fault_time']
    del final_test_b_data['msg']

    all_value = rawdata['words'].append(final_test_a_data['words']).append(final_test_b_data['words'])
    all_value.to_csv('../user_data/alldata.csv', index=False)


# In[4]:
print('alldata starting ')

alldata_preprocess()
print('alldata end ')

from tokenizers import Tokenizer
from tokenizers.models import BPE
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
from tokenizers.trainers import BpeTrainer
#加入一些特殊字符
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],vocab_size=Vocab_Size)

#空格分词器
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()
print('tokenizer starting ')
#保存语料库文件
tokenizer.train(['../user_data/alldata.csv'], trainer)
tokenizer.mask_token='[MASK]'
tokenizer.save("../user_data/tokenizer-my-Whitespace.json")
print('tokenizer end ')