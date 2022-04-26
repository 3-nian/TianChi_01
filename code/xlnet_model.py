from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from transformers import XLNetForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold
from config import *


def data_preprocess():
    rawdata = pd.read_csv(train_data_file, encoding='UTF-8', names=["sn", "fault_time", "msg", "label"])
    rawdata['words'] = rawdata['msg'].str.replace(r'|', ' ')
    del rawdata['msg']
    del rawdata['sn']
    del rawdata['fault_time']
    rawdata['label'] = rawdata['label'].astype(int)
    # 数据划分
    # 如果之前已经做了就直接加载
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
    # 预测
    f = pd.read_csv(final_test_a_data_file, encoding='UTF-8', names=["sn", "fault_time", "msg"])
    f['words'] = f['msg'].str.replace(r'|', ' ')
    del f['msg']
    del f['sn']
    del f['fault_time']
    f.to_csv(USER_DATA_PATH + 'final_test_data.csv', index=False)
    return rawdata


# 注意这里用了另外一种方式加载Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=USER_DATA_PATH + "tokenizer-my-Whitespace.json")
tokenizer.mask_token = '[MASK]'
tokenizer.pad_token = '[PAD]'


def preprocess_function(examples):
    return tokenizer(examples['words'], truncation=True, max_length=Max_len, padding='max_length')


model_checkpoint = USER_DATA_PATH + "XLNet"  # 所选择的预训练模型
model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


def get_arg(num, train=True):
    train_args = TrainingArguments(
        USER_DATA_PATH + str(num) + "_test-glue",
        evaluation_strategy="epoch",  # 每个epcoh会做一次验证评估；
        save_strategy="epoch",
        # logging_dir='test-glue/log',
        # logging_strategy="epoch",
        # report_to="tensorboard",
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=num,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,  # 根据哪个评价指标选最优模型
        save_steps=200,
        save_total_limit=2,
        dataloader_drop_last=True
    )

    test_args = TrainingArguments(
        USER_DATA_PATH + str(num) + "_test-glue",
        evaluation_strategy="epoch",  # 每个epcoh会做一次验证评估；
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=num,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,  # 根据哪个评价指标选最优模型
        save_steps=10_000,
        save_total_limit=2,
        no_cuda=True
    )
    return train_args if train else test_args


# 10折交叉验证
kf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
raw_data = data_preprocess()
if LOCAL_TEST:
    raw_data = raw_data.loc[:100, :]
y = raw_data['label']
train_loss = []
test_f1 = []
best_list = []

X_train, X_test, y_train, y_test = train_test_split(raw_data, y, test_size=0.05, stratify=y)

# 测试集
X_test.columns = ['label', 'words']
X_test.to_csv(USER_DATA_PATH + 'test_data.csv', index=False)
X_train = pd.DataFrame(X_train)
for epoch in Num_epoch:
    best_f1 = 0.0
    for k, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        train_data, test_data = X_train.iloc[train_index], X_train.iloc[test_index]

        # 训练集
        train_data.columns = ['label', 'words']
        train_data.to_csv(USER_DATA_PATH + 'train_data.csv', index=False)
        # 开发集
        test_data.columns = ['label', 'words']
        test_data.to_csv(USER_DATA_PATH + 'dev_data.csv', index=False)

        dataset = load_dataset('csv', data_files={'train': USER_DATA_PATH + 'train_data.csv',
                                                  'dev': USER_DATA_PATH + 'dev_data.csv',
                                                  'test': USER_DATA_PATH + 'test_data.csv'})
        encoded_dataset = dataset.map(preprocess_function, batched=True)

        model_checkpoint = USER_DATA_PATH + "XLNet" if k == 0 else USER_DATA_PATH + str(epoch) + "_test-glue"  # 所选择的预训练模型
        model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

        trainer = Trainer(
            model,
            get_arg(epoch),
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["dev"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        train_result = trainer.train()
        loss_train = train_result.training_loss
        train_loss.append(loss_train)

        trainer = Trainer(
            model,
            get_arg(epoch, train=False),
            eval_dataset=encoded_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        score_test = trainer.evaluate()['eval_f1']
        test_f1.append(score_test)
        best_list.append(best_f1)
        if score_test > best_f1:
            best_f1 = score_test
            trainer.save_model(USER_DATA_PATH + str(epoch) + "_test-glue")
        print(
            '  - {fold:4} fold train loss: {loss_train: 8.5f}, best F1: {best_f1:8.5f} , test F1: {score_test:8.5f}, '.format(
                fold=k + 1, loss_train=loss_train,
                best_f1=best_f1, score_test=score_test))
