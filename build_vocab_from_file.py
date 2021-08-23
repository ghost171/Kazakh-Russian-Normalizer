from build_vocab_for_model import build_vocab, save_vocab
import torch
import numpy as np
import bert
from bert import tokenization
import tensorflow as tf
from get_model_for_tokenization import get_model
from numba import cuda
import torch


USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

max_seq_length = 10
labse_model, labse_layer = get_model(model_url="https://tfhub.dev/google/LaBSE/1", max_seq_length=max_seq_length)

device = cuda.get_current_device()
device.reset()

vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()

with tf.device('/cpu:0'):
    do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
    tokenizer = bert.tokenization.FullTokenizer(vocab_file, do_lower_case)

vocab_unnormalized = build_vocab('./data/train.ut', '/home/ghost/Annotated_2/Scriptur_task/unnormalized/', 'ut')
vocab_normalized = build_vocab('./data/train.nt', '/home/ghost/Annotated_2/Scriptur_task/normalized/', 'nt')

print("VOCAB")
save_vocab(vocab_normalized, 'vocab_unnormalized/vocab_unnormalized.csv')
print("VOCAB")
save_vocab(vocab_normalized, 'vocab_normalized/vocab_normalized.csv')
