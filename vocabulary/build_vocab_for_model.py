import io
import csv
from collections import Counter
import tensorflow as tf
import torch
import numpy as np

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


def encode(input_text, tokenizer, labse_model, max_seq_length=64):
    input_ids, input_mask, segment_ids = create_input(input_text, tokenizer, max_seq_length)
    return labse_model([input_ids, input_mask, segment_ids])


def tokenize_sentences(text, tokenizer):
    return [tok for tok in tokenizer.tokenize(text)]

def build_vocab(path_of_dataset, data_folder, ext, tokenizer):
    #counter = Counter()
    all_dataset_vocab = []
    index_for_vocab = 0
    all_dataset_vocab = torch.tensor(all_dataset_vocab)
    all_dataset_vocab = all_dataset_vocab.to('cuda')

    with io.open(path_of_dataset, encoding="utf8") as f:
        print("BUILD VOCAB 1")
        for i, string_ in enumerate(f):
            #print("PRINT SAMPLES")
            #name = "sample_" + str("%10.7o"% i) + '.' + ext
            #with open(data_folder + name, 'w') as f:
            #    f.write(string_)

            tokenized = tokenize_sentences(string_, tokenizer) 

            #vocab_file = open(path_for_recording_vocab, "w")
            #writer = csv.writer(vocab_file)
            print("TOKENIZED", tokenized)
            print(tokenized)
            embedded = encode(tokenized, tokenizer)
            print(embedded)
            tokenized = torch.tensor(embedded)
            tokenized = tokenized.to('cuda')
            for value in tokenized:
                if value not in all_dataset_vocab:
                    index_for_vocab += 1
                    all_dataset_vocab.append([index_for_vocab, value])

    return all_dataset_vocab

def save_vocab(vocab, path):
    with io.open(path, encoding="utf8") as f:
        writer = csv.writer(f)
        for key, value in vocab:
            writer.writerow([key, value])
        