import torch
import torch.nn as nn
import math, copy, time
import numpy as np
from  torch.utils.data import DataLoader
import tensorflow_hub as hub
import tensorflow as tf
from numba import cuda
import pandas as pd
import bert
from numba import cuda
from torchtext import data
from bert import tokenization
import sacrebleu
import io
from model import   greedy_decode,  lookup_words, print_examples, EncoderDecoder, Encoder, rebatch, Decoder
from model import   BahdanauAttention, Batch, run_epoch, SimpleLossCompute, train, print_data_info, Generator
from model import make_model
from Dataset import Collation, CustomTextDataset

def get_model(model_url, max_seq_length):
    with tf.device('/cpu:0'):
        labse_layer = hub.KerasLayer(model_url, trainable=False)

      # Define input.
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                             name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                     name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                          name="segment_ids")

      # LaBSE layer.
        pooled_output,  _ = labse_layer([input_word_ids, input_mask, segment_ids])

      # The embedding is l2 normalized.
        pooled_output = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

      # Define model.
        labse_model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids],
        outputs=pooled_output)
    return labse_model, labse_layer

def tokenize_kz_unnormalized(text):
    out = [tok for tok in ininormer.tokenize(text)]
    return out

def tokenize_kz_normalized(text):
    out = [tok for tok in ininormer.tokenize(text)]
    return out

# we will use CUDA if it is available
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

import tensorflow.compat.v1 as tf

with tf.device('/cpu:0'):
    do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
    tokenizer = bert.tokenization.FullTokenizer(vocab_file, do_lower_case)


import tensorflow as tf

'''def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return build_vocab_from_iterator(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])'''

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
LOWER = True

ininormer = tokenizer

SRC = data.Field(tokenize=tokenize_kz_unnormalized, batch_first=True, lower=LOWER, include_lengths=True, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)

TRG = data.Field(tokenize=tokenize_kz_normalized, batch_first=True, lower=LOWER, include_lengths=True, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

MAX_LEN=25

def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer.tokenize(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

vocab = build_vocab('train.ut', ininormer)

def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()

save_vocab(vocab, 'output.txt')


print("VOCAB_FILE", vocab_file)
src_dataset = CustomTextDataset(vocab_file, 'train.ut', './')
trg_dataset = CustomTextDataset(vocab_file, 'train.nt', './')

#collate_fn = Collation(vocab['<pad>'], vocab['<pad>'], vocab['<pad>'])


#src_dataloader = DataLoader(src_dataset, batch_size=5, collate_fn=collate_fn)
'''trg_dataloader = DataLoader(trg_dataset, batch_size=5, collate_fn=collate_fn)'''



from torchtext.datasets import TranslationDataset

text_dataset = TranslationDataset(path='./', exts=('train.ut', 'train.nt'), fields=(SRC, TRG))

from torchtext import data, datasets

if True:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    LOWER = True

    # we include lengths to provide to the RNNs
    SRC = data.Field(tokenize=tokenize_kz_unnormalized,
                     batch_first=True, lower=LOWER, include_lengths=True,
                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)

    TRG = data.Field(tokenize=tokenize_kz_normalized,
                     batch_first=True, lower=LOWER, include_lengths=True,
                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

    MAX_LEN = 25  # NOTE: we filter out a lot of sentences for speed
    train_data, valid_data, test_data = text_dataset.splits(path='./', exts=('.ut', '.nt'), fields=(SRC, TRG), filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 5  # NOTE: we limit the vocabulary to frequent words for speed
    SRC.build_vocab(text_dataset.src, min_freq=MIN_FREQ)
    TRG.build_vocab(text_dataset.trg, min_freq=MIN_FREQ)

    PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]

print_data_info(train_data, SRC, TRG)

train_iter = data.BucketIterator(train_data, batch_size=1, train=True, sort_within_batch=True,
                                 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                 device=DEVICE)


device = cuda.get_current_device()
device.reset()

valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False,
                           device=DEVICE)

model = make_model(len(SRC.vocab), len(TRG.vocab),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)

dev_perplexities = train(model, PAD_INDEX, train_iter, valid_iter, SRC, TRG, print_every=10)

hypotheses = []
alphas = []
for batch in valid_iter:
    batch = rebatch(PAD_INDEX, batch)
    pred, attention = greedy_decode(
    model, batch.src, batch.src_mask, batch.src_lengths, max_len=25,
        sos_index=TRG.vocab.stoi[SOS_TOKEN],
        eos_index=TRG.vocab.stoi[EOS_TOKEN])
    hypotheses.append(pred)
    alphas.append(attention)

hypotheses = [lookup_words(x, TRG.vocab) for x in hypotheses]

hypotheses = [" ".join(x) for x in hypotheses]


idx = 2
src = valid_data[idx].src + ["</s>"]
trg = valid_data[idx].trg + ["</s>"]
pred = hypotheses[idx].split() + ["</s>"]
pred_att = alphas[idx][0].T[:, :len(pred)]
print("src", src)
print("ref", trg)
print("pred", pred)




