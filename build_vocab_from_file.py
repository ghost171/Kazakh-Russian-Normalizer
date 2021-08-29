
import torch
import numpy as np
import bert
from bert import tokenization
import tensorflow as tf
from get_model_for_tokenization import get_model
from numba import cuda
import torch
import io
import csv

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


def tokenize_sentences(text, tokenizer):
    return [tok for tok in tokenizer.tokenize(text)]

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
            if i == 3:
                break
            tokenized = tokenize_sentences(string_, tokenizer) 

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
            #embedded_tensor = embedded_tensor.to('cuda')
            for i, value in enumerate(tokenized):
                print("ALL DATASET VOCAB")
                #print(all_dataset_vocab)
                print("ALL DATASET VOCAB")
                if value not in values:
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

#with tf.device('/cpu:0'):
do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert.tokenization.FullTokenizer(vocab_file, do_lower_case)

print("BUILD VOCAB")
vocab_unnormalized, vocab_unnormalized_embedded = build_vocab('./data/train.ut', '/home/ghost17/Annotated_3/Kazakh-Russian-Normalizer/unnormalized/', 'ut', tokenizer, labse_model)
vocab_normalized, vocab_normalized_embedded = build_vocab('./data/train.nt', '/home/ghost17/Annotated_3/Kazakh-Russian-Normalizer/normalized/', 'nt', tokenizer, labse_model)
print("BUILD VOCAB")
#print("NORMALIZED", vocab_normalized)
#print("UNNORMALIZED", vocab_unnormalized)

print("VOCAB")
save_vocab(vocab_unnormalized, 'vocab_unnormalized/vocab_unnormalized.csv')
print("VOCAB")
save_vocab(vocab_unnormalized_embedded, 'vocab_unnormalized/vocab_unnormalized_embedded.csv')
print("VOCAB")
save_vocab(vocab_normalized, 'vocab_normalized/vocab_normalized.csv')
print("VOCAB")
save_vocab(vocab_normalized_embedded, 'vocab_normalized/vocab_normalized_embedded.csv')

