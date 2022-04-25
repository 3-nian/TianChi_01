from config import *
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaConfig, AlbertConfig, XLNetConfig
from transformers import RobertaForMaskedLM, AlbertForMaskedLM, XLNetLMHeadModel
from transformers import Trainer, TrainingArguments

# 注意这里用了另外一种方式加载Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=USER_DATA_PATH + "tokenizer-my-Whitespace.json")
tokenizer.mask_token = '[MASK]'
tokenizer.pad_token = '[PAD]'
# 数据预处理
dataset = load_dataset('csv', data_files={'train': USER_DATA_PATH + 'alldata.csv'})


def preprocess_function(examples):
    return tokenizer(examples['words'], truncation=True, max_length=Max_len, padding='max_length')


encoded_dataset = dataset.map(preprocess_function, batched=True)
# 数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    # mlm表示是否使用masked language model；mlm_probability表示mask的几率
)

# 模型配置
# 自己修改部分配置参数
config_kwargs = {
    "d_model": 512,
    "n_head": 4,
    "vocab_size": 2048,  # tokenizer.vocab_size, # 自己设置词汇大小
    "bi_data": False,
    "n_layer": 8
}
# 将模型的配置参数载入
'''RobertaConfig,AlbertConfig,XLNetConfig'''
config = XLNetConfig(**config_kwargs)
# 载入预训练模型，这里其实是根据某个模型结构调整config然后创建模型
'''RobertaForMaskedLM,AlbertForMaskedLM,XLNetLMHeadModel'''
model = XLNetLMHeadModel(config=config)

training_args = TrainingArguments(
    output_dir=USER_DATA_PATH + "XLNet",
    overwrite_output_dir=True,
    num_train_epochs=5,  # 训练epoch次数
    per_device_train_batch_size=BATCH_SIZE,  # 训练时的batchsize
    save_steps=200,  # 每200步保存一次模型
    save_total_limit=2,  # 最多保存两次模型
    prediction_loss_only=True,
    dataloader_drop_last=True,
    logging_dir=USER_DATA_PATH + 'XLNet/log',
    logging_strategy="epoch",
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,  # 数据收集器在这里
    train_dataset=encoded_dataset["train"]  # 注意这里选择的是预处理后的数据集
)
print('trainer pretraining starting ')
# 开始训练
trainer.train()
# #保存模型
trainer.save_model(USER_DATA_PATH + "XLNet")
print('trainer pretraining end ')
