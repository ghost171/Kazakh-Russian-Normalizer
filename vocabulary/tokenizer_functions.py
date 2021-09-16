from vocabulary.get_model_for_tokenization import get_model
import torch
import numpy as np
import bert

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

#device = cuda.get_current_device()
#device.reset()

def tokenize_kz_unnormalized(text):
    out = [tok for tok in ininormer.tokenize(text)]
    return out

def tokenize_kz_normalized(text):
    out = [tok for tok in ininormer.tokenize(text)]
    return out

vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()

import tensorflow.compat.v1 as tf

with tf.device('/cpu:0'):
    do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
    tokenizer = bert.tokenization.FullTokenizer(vocab_file, do_lower_case)

ininormer = tokenizer
