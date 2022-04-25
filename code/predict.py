import os
from config import *
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from transformers import XLNetForSequenceClassification, TrainingArguments, Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 注意这里用了另外一种方式加载Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=USER_DATA_PATH + "tokenizer-my-Whitespace.json")
tokenizer.mask_token = '[MASK]'
tokenizer.pad_token = '[PAD]'


def preprocess_function(examples):
    return tokenizer(examples['words'], truncation=True, max_length=Max_len, padding='max_length')


model_checkpoint = USER_DATA_PATH + "20_test-glue"  # "BERT" #所选择的预训练模型

model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

test_args = TrainingArguments(
    USER_DATA_PATH + "20_test-glue",
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    metric_for_best_model=metric_name  # 根据哪个评价指标选最优模型
)

dataset = load_dataset('csv', data_files={'train': USER_DATA_PATH + 'train_data.csv',
                                          'dev': USER_DATA_PATH + 'dev_data.csv',
                                          'test': USER_DATA_PATH + 'test_data.csv'})  # 这里建议用完全路径，否则可能卡住
encoded_dataset = dataset.map(preprocess_function, batched=True)

trainer = Trainer(
    model,
    test_args,
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print(trainer.evaluate())

print('************predict starting********************')
final_test_dataset = load_dataset('csv', data_files={
    'res': USER_DATA_PATH + 'final_test_data.csv'})  # ,cache_dir='res-fine-tune'
encoded_final_test_dataset = final_test_dataset.map(preprocess_function, batched=True)
res = trainer.predict(test_dataset=encoded_final_test_dataset["res"])
csv = pd.DataFrame(np.argmax(res[0], 1), columns=['label'])
print('************predict end********************')
test_data = pd.read_csv(FEATURE_PATH + 'names_test16_a.csv', header=None, names=['sn', 'fault_time', 'msg'])
test_data['label'] = csv
submit_data = pd.read_csv(TC_DATA_PATH + 'final_submit_dataset_a.csv')
submit_final_data = pd.merge(submit_data, test_data, on=['sn', 'fault_time'], how='left')
submit_final_data.sort_values(by=['sn', 'fault_time'], inplace=True)
submit_final_data['label'].fillna(method='ffill', axis=0, inplace=True)
submit_final_data.drop(['msg'], axis=1, inplace=True)
submit_final_data.to_csv(PREDICT_PATH + 'predictions.csv', index=False)
