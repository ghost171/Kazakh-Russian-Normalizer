import numpy as np
import torch
from  torch.utils.data import DataLoader
import io
from model import train
from model import make_model
from dataset_functions import Collation, CustomTextDataset

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

'''max_seq_length = 10
labse_model, labse_layer = get_model(model_url="https://tfhub.dev/google/LaBSE/1", max_seq_length=max_seq_length)

device = cuda.get_current_device()
device.reset()

vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()

with tf.device('/cpu:0'):
    do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
    tokenizer = bert.tokenization.FullTokenizer(vocab_file, do_lower_case)

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
LOWER = True

ininormer = tokenizer

UNNORMALIZED = data.Field(tokenize=tokenize_kz_unnormalized, batch_first=True, lower=LOWER, include_lengths=True, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)

NORMALIZED = data.Field(tokenize=tokenize_kz_normalized, batch_first=True, lower=LOWER, include_lengths=True, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

MAX_LEN=25

vocab_unnormalized = build_vocab('./data/train.ut', ininormer, '/home/ghost/Annotated_2/Scriptur_task/unnormalized/', 'ut')
vocab_normalized = build_vocab('./data/train.nt', ininormer, '/home/ghost/Annotated_2/Scriptur_task/normalized/', 'nt')'''


def load_vocab(path):
    vocab = dict()
    with io.open(filepath, encoding="utf8") as f:
        for i, string_ in enumerate(f):
            vocab[i] = string_
    return vocab

vocab_unnormalized = load_vocab('vocab_unnormalized/unnormalized_vocab.csv')
vocab_normalized = load_vocab('vocab_normalized/normalized_vocab.csv')

#print("VOCAB_FILE", vocab_file)
src_dataset = CustomTextDataset(vocab, 'train.ut', './data/')
trg_dataset = CustomTextDataset(vocab, 'train.nt', './data/')

#df = pd.DataFrame.from_dict(vocab, orient='index').reset_index()
#compression_opts = dict(method='zip', archive_name='vocab.csv')
#df.to_csv('vocab_unnormalized/vocab.zip', index=False, compression=compression_opts)

collate_fn = Collation(vocab['<pad>'], vocab['<pad>'], vocab['<pad>'])


src_dataloader = DataLoader(src_dataset, batch_size=5, collate_fn=collate_fn)
trg_dataloader = DataLoader(trg_dataset, batch_size=5, collate_fn=collate_fn)

#train_iter = data.BucketIterator(src_dataloader, batch_size=5, train=True, sort_within_batch=True,
#                                 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False, 
#                                 device=DEVICE)

#device = cuda.get_current_device()
#device.reset()

model = make_model(len(UNNORMALIZED.vocab), len(NORMALIZED.vocab),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)

src_dataset_valid = CustomTextDataset(vocab, 'train.ut', './data/')
trg_dataset_valid = CustomTextDataset(vocab, 'train.nt', './data/')


dev_perplexities = train(model, PAD_INDEX, train_iter, valid_iter, UNNORMALIZED, NORMALIZED, print_every=10)


'''from torchtext.datasets import TranslationDataset

text_dataset = TranslationDataset(path='./data/', exts=('train.ut', 'train.nt'), fields=(UNNORMALIZED, NORMALIZED))

from torchtext import data, datasets

if True:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    LOWER = True

    # we include lengths to provide to the RNNs
    UNNORMALIZED = data.Field(tokenize=tokenize_kz_unnormalized,
                     batch_first=True, lower=LOWER, include_lengths=True,
                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)

    NORMALIZED = data.Field(tokenize=tokenize_kz_normalized,
                     batch_first=True, lower=LOWER, include_lengths=True,
                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

    MAX_LEN = 25  # NOTE: we filter out a lot of sentences for speed
    train_data, valid_data, test_data = text_dataset.splits(path='./data/', exts=('.ut', '.nt'), fields=(UNNORMALIZED, NORMALIZED), filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 5  # NOTE: we limit the vocabulary to frequent words for speed
    UNNORMALIZED.build_vocab(text_dataset.src, min_freq=MIN_FREQ)
    NORMALIZED.build_vocab(text_dataset.trg, min_freq=MIN_FREQ)

    PAD_INDEX = NORMALIZED.vocab.stoi[PAD_TOKEN]

print_data_info(train_data, UNNORMALIZED, NORMALIZED)

train_iter = data.BucketIterator(train_data, batch_size=1, train=True, sort_within_batch=True,
                                 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                 device=DEVICE)


device = cuda.get_current_device()
device.reset()

valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False,
                           device=DEVICE)

model = make_model(len(UNNORMALIZED.vocab), len(NORMALIZED.vocab),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)

dev_perplexities = train(model, PAD_INDEX, train_iter, valid_iter, UNNORMALIZED, NORMALIZED, print_every=10)

hypotheses = []
alphas = []
for batch in valid_iter:
    batch = rebatch(PAD_INDEX, batch)
    pred, attention = greedy_decode(
    model, batch.src, batch.src_mask, batch.src_lengths, max_len=25,
        sos_index=NORMALIZED.vocab.stoi[SOS_TOKEN],
        eos_index=NORMALIZED.vocab.stoi[EOS_TOKEN])
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
print("pred", pred)'''




