import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import torch
import bert

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

def encode(input_text, tokenizer, labse_model, max_seq_length=60):
    input_ids, input_mask, segment_ids = create_input(input_text, tokenizer, max_seq_length)
    return labse_model([input_ids, input_mask, segment_ids])

def get_LABSE(model_url, max_seq_length):
    labse_layer = hub.KerasLayer(model_url, trainable=False)

    # Define input.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                             name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                     name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                          name="segment_ids")

    #LaBSE layer.
    pooled_output, _ = labse_layer([input_word_ids, input_mask, segment_ids])

    # The embedding is l2 normalized.
    pooled_output = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

    # Define model.
    labse_model = tf.keras.Model(
    inputs=[input_word_ids, input_mask, segment_ids],
    outputs=pooled_output)
    return labse_model, labse_layer

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda') # or set to 'cpu'
print("USE_CUDA", USE_CUDA)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

max_seq_length = 60
labse_model, labse_layer = get_LABSE(model_url="https://tfhub.dev/google/LaBSE/1", max_seq_length=max_seq_length)

vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert.tokenization.FullTokenizer(vocab_file, do_lower_case)
print('Loaded models!')
df_train = pd.read_csv('filelists/train_filelists.csv')
df_val = pd.read_csv('filelists/val_filelists.csv')
df_test = pd.read_csv('filelists/test_filelists.csv')

df = pd.concat([df_train, df_val, df_test])

filenames = df['filenames'].to_list()
#unnormalized texts
data_label = 'unnormalized'
input_folder_name = f'data/raw/{data_label}/'
tokenized_folder_name = f'data/tokenized/{data_label}/'
embedded_folder_name = f'data/embedded/{data_label}/'

if not os.path.exists(tokenized_folder_name):
    os.makedirs(tokenized_folder_name)

if not os.path.exists(embedded_folder_name):
    os.makedirs(embedded_folder_name)

vocab_unnormalized = set()
# iteration = 0

for filename in filenames:
    # iteration += 1
    with open(input_folder_name + filename, 'r') as f:
        sentence = f.readline()
    # if iteration == 707:
    #     print("SENTENCE 707            ", sentence)
    # if iteration == 2203 or iteration==2202 or (iteration >= 2206 and iteration <=2220):
    #     print("SENTENCE ", sentence)
    tokens = tokenizer.tokenize(sentence)
    embeddings = encode(tokens, tokenizer, labse_model)
    with open(tokenized_folder_name + filename, 'w') as f:
        [f.write(token) and f.write('\n') for token in tokens]
        f.close()

    np.savetxt(embedded_folder_name + filename, embeddings)
    #with open(embedded_folder_name + filename, 'w') as f:
       #[f.write(embedding.numpy()) for embedding in embeddings]
       #f.close()
    # print("ITERATION!          ", iteration)
    vocab_unnormalized = vocab_unnormalized.union(set(tokens))


iteration = 0
data_label = 'normalized'
input_folder_name = f'data/raw/{data_label}/'
tokenized_folder_name = f'data/tokenized/{data_label}/'
embedded_folder_name = f'data/embedded/{data_label}/'

vocab_normalized = set()

iteration = 0
for filename in filenames:
    iteration += 1
    with open(input_folder_name + filename, 'r') as f:
        sentence = f.readline()

    tokens = tokenizer.tokenize(sentence)
    embeddings = encode(tokens, tokenizer, labse_model)
    with open(tokenized_folder_name + filename, 'w') as f:
        [f.write(token) and f.write('\n') for token in tokens]
        f.close()

    np.savetxt(embedded_folder_name + filename, embeddings)
    #with open(embedded_folder_name + filename, 'w') as f:
    #   [f.write(embedding.numpy()) for embedding in embeddings]
    #   f.close()
    #
    #
    # print("ITERATION 2!           ", iteration)
    vocab_normalized = vocab_normalized.union(set(tokens))

vocab_unnormalized = [(i, value) for i, value in enumerate(vocab_unnormalized)]
vocab_normalized = [(i, value) for i, value in enumerate(vocab_normalized)]


if not os.path.exists('data/vocab_unnormalized/'):
    os.makedirs('data/vocab_unnormalized/')

if not os.path.exists('data/vocab_normalized/'):
    os.makedirs('data/vocab_normalized/')

with open('data/vocab_unnormalized/vocab_unnormalized.txt', 'w') as f:
    for token in vocab_unnormalized:
        [f.write(str(token[0]) + ', ' + str(token[1]) + "\n")]

with open('data/vocab_normalized/vocab_normalized.txt', 'w') as f:
    for token in vocab_normalized:
        print("TOKEN", token[0])
        [f.write(str(token[0]) + ", " + str(token[1]) + "\n")]

# print("WRITING")
#
# print("VOCAB UNNORMALIZED")
# print(vocab_unnormalized)
# print("VOCAB UNNORMALIZED")
# print("VOCAB NORMALIZED")
# print(vocab_normalized)
# print("VOCAB NORMALIZED")
