import io
import csv

import torch
import numpy as np
import pandas as pd

import bert
from bert import tokenization

import tensorflow as tf
import tensorflow_hub as hub
from numba import cuda


def get_model(model_url, max_seq_length):
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


def create_input(input_strings, tokenizer, max_seq_length):

    input_ids_all, input_mask_all, segment_ids_all = [], [], []
    for input_string in input_strings:
        # Tokenize input.
        input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        sequence_length = min(len(input_ids), max_seq_length)

        # Padding or truncation.
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[:max_seq_length]
        else:
            input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

        input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

        input_ids_all.append(input_ids)
        input_mask_all.append(input_mask)
        segment_ids_all.append([0] * max_seq_length)

    return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)


def encode(input_text, tokenizer, labse_model, max_seq_length=10):
    input_ids, input_mask, segment_ids = create_input(input_text, tokenizer, max_seq_length)
    return labse_model([input_ids, input_mask, segment_ids])


def build_vocab(path_of_dataset, data_folder, ext, tokenizer, labse_model):
    #counter = Counter()
    all_dataset_vocab = []
    index_for_vocab = 0
    all_dataset_vocab_embedded = []
    values = []
    #all_dataset_vocab = torch.tensor(all_dataset_vocab)
    #all_dataset_vocab.to('cuda')

    with io.open(path_of_dataset, encoding="utf8") as f:
        print("BUILD VOCAB 1")
        for i, string_ in enumerate(f):
            #print("PRINT SAMPLES")
            #name = "sample_" + str("%10.7o"% i) + '.' + ext
            #with open(data_folder + name, 'w') as f:
            #    f.write(string_)
            tokenized = [tok for tok in tokenizer.tokenize(string_)]

            #vocab_file = open(path_for_recording_vocab, "w")
            #writer = csv.writer(vocab_file)
            #print("TOKENIZED", tokenized)
            #print(tokenized)
            embedded = encode(tokenized, tokenizer, labse_model)
            #embedded_tensor = tf.convert_to_tensor(embedded.numpy(), dtype=None, dtype_hint=None, name=None)
            #print("EMBEDDED", embedded)
            #print(type(embedded))
            embedded = embedded.numpy()
            #print(type(embedded))
            embedded = torch.Tensor(embedded)
            embedded.to('cuda')
            #tokenized = torch.Tensor(tokenized)
            #@tokenized.to('cuda')
            for i, value in enumerate(embedded):
                print("ALL DATASET VOCAB")
                #print(all_dataset_vocab)
                print("ALL DATASET VOCAB")
                #if value not in values:
                
                print("ALL DATASET VOCAB", all_dataset_vocab)
                index_for_vocab += 1
                all_dataset_vocab.append([index_for_vocab, value])
                all_dataset_vocab_embedded.append([index_for_vocab, embedded[i]])
                values.append(value)

    return all_dataset_vocab, all_dataset_vocab_embedded


def save_vocab(vocab, path):
    with io.open(path, 'w', encoding="utf8") as f:
        writer = csv.writer(f)
        for key, value in vocab:
            writer.writerow([key, value])

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0') # or set to 'cpu'

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

max_seq_length = 10
labse_model, labse_layer = get_model(model_url="https://tfhub.dev/google/LaBSE/1", max_seq_length=max_seq_length)

device = cuda.get_current_device()
device.reset()

vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert.tokenization.FullTokenizer(vocab_file, do_lower_case)

vocab_unnormalized, vocab_unnormalized_embedded = build_vocab('./data/train.ut', '/home/ghost17/Annotated_3/Kazakh-Russian-Normalizer/unnormalized/', 'ut', tokenizer, labse_model)
vocab_normalized, vocab_normalized_embedded = build_vocab('./data/train.nt', '/home/ghost17/Annotated_3/Kazakh-Russian-Normalizer/normalized/', 'nt', tokenizer, labse_model)

save_vocab(vocab_unnormalized, 'vocab_unnormalized/vocab_unnormalized.csv')
save_vocab(vocab_unnormalized_embedded, 'vocab_unnormalized/vocab_unnormalized_embedded.csv')
save_vocab(vocab_normalized, 'vocab_normalized/vocab_normalized.csv')
save_vocab(vocab_normalized_embedded, 'vocab_normalized/vocab_normalized_embedded.csv')

