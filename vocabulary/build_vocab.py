import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # SET TO 0

import pandas as pd
import numpy as np

import torch
from transformers import BertTokenizer, BertModel

device = 'cuda' # or set to 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased').eval().to(device)


def get_embeddings(text, bert_model, tokenizer, device):

    marked_text = tokenizer.decode(tokenizer(text).input_ids)
    tokenized_text = tokenizer.tokenize(marked_text)
    #print(f'The length of tokens is {len(tokenized_text)}')
    segments_ids = [1] * len(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    with torch.no_grad():
        outputs = bert_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    token_vecs_sum = [] # final embeddings from BERT

    for token in token_embeddings.transpose(1, 0):
        sum_vec = torch.sum(token[-4:], dim=0).to(device)
        token_vecs_sum.append(sum_vec)

    return tokenized_text, token_vecs_sum

text = 'Сәлем, сенің атың кім?'

tokenized_text, embeddings = get_embeddings(text, model, tokenizer, device)

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
    tokens, embeddings = get_embeddings(text, model, tokenizer, device)

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
