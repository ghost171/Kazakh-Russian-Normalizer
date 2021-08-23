from build_vocab_for_model import build_vocab, save_vocab
import torch
import numpy as np
import bert
from bert import tokenization
import tensorflow as tf
from get_model_for_tokenization import get_model
from numba import cuda
import torch
import io


def default_data_creator(filepath, data_folder, ext):
    #counter = Counter()
    #answer = dict()
    with io.open(filepath, encoding="utf8") as f:
        for i, string_ in enumerate(f):

            name = "sample_" + str("%10.7o"% i) + '.' + ext
            with open(data_folder + name, 'w') as file:
                file.write(string_)
            #answer[i] = string_

    #return answer

def tokenized_data_creator(filepath, data_folder, ext, ininormer):
    with io.open(filepath, encoding="utf8") as f:
        for i, string_ in enumerate(f):
            name = "sample_" + str("%10.7o" % i) + '.' + ext
            with open(data_folder + name, 'w') as file:
                file.write(ininormer(string_))

default_data_creator('./data/train.ut', '/home/ghost/Annotated_2/Scriptur_task/data_splitted_to_sentences/row/unnormalized/', 'ut')
default_data_creator('./data/train.nt', '/home/ghost/Annotated_2/Scriptur_task/data_splitted_to_sentences/row/normalized/', 'nt')

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

tokenized_data_creator('./data/train.ut', '/home/ghost/Annotated_2/Scriptur_task/data_splitted_to_sentences/tokenized/unnormalized/', 'ut', tokenizer)
default_data_creator('./data/train.nt', '/home/ghost/Annotated_2/Scriptur_task/data_splitted_to_sentences/tokenized/normalized/', 'nt', tokenizer)

