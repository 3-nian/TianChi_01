import torch
dropout = 0.1
train_data_file = '../feature/names_train16.csv'
test_index_file='../feature/test_index.pkl'
train_index_file='../feature/train_index.pkl'
final_test_a_data_file = '../feature/names_test16_a.csv'
final_test_b_data_file = '../feature/names_test16_b.csv'
Max_len = 256
Num_epoch = [5]
BATCH_SIZE = 32

Vocab_Size= 3000

gpu=0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
