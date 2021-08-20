import io
from collections import Counter

def build_vocab(filepath, tokenizer, data_folder, ext):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for i, string_ in enumerate(f):
            name = "sample_" + str("%10.7o"% i) + '.' + ext
            with open(data_folder + name, 'w') as f:
                f.write(string_)
            counter.update(tokenizer.tokenize(string_))
    return counter
