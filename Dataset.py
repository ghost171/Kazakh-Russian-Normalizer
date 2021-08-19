from torch.utils.data import Dataset
from collections import Counter
from torchtext.vocab import Vocab

class CustomTextDataset(Dataset):
    def __init__(self, vocab, split_csv_path, text_dir):
        self.vocab = vocab
        self.info = pd.read_csv(split_csv_path)
        self.text_dir = text_dir

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        unnormalized_text = os.path.join(self.text_dir, self.info.iloc[idx, 0])
        normalized_text = os.path.join(self.text_dir, self.info.iloc[idx, 1])

        unnormalized_text = torch.from_numpy(np.array([self.vocab[token] for token in unnormalized_text]),
                                             dtype=torch.int32)  # (1D)
        normalized_text = torch.from_numpy(np.array([self.vocab[token] for token in normalized_text]),
                                           dtype=torch.int32)  # chekp shape, it should be compatible with model input shape
        # for model input I suppose [B, MAX_LEN_AMONG_BATCH] CHECK SRC torch.Size([1, 15])!
        # for model output ([1, 15, 256])
        return unnormalized_text, normalized_text


class Collation():
    def __init__(self, init_token, eos_token, pad_token):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token  # ids from vocab

    def __call__(self, batch):
        """
        batch: list of size batch_size where each element is a tuple of unnormalized_text: torch.int32 and normalized_text: torch.int32].
        """
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size for x in batch]),  ## check .size !!! maybe len
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        unnormalized_text_padded = torch.LongTensor(len(batch),
                                                    max_input_len)  # depends on model input data type, CHECK!!!
        unnormalized_text_padded.zero_()  # FILL WITH PAD_TOKEN ID

        # IF NEEDED PREPEND AND APPEND CORRESPONDING init_token and eos_token OR DO IT IN __get_item__()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            unnormalized_text_padded[i, :text.size(0)] = text

        output_lengths, _ = torch.sort(torch.LongTensor([len(x[1]) for x in batch]), dim=0, descending=True)
        max_output_len = output_lengths[0]

        normalized_text_padded = torch.LongTensor(len(batch), max_input_len)
        normalized_text_padded.zero_()


        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][1]
            normalized_text_padded[i, :text.size(0)] = text

        return (unnormalized_text_padded, normalized_text_padded)