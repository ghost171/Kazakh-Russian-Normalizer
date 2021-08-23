import io
from collections import Counter

<<<<<<< HEAD
def build_vocab(filepath, tokenizer, data_folder, ext):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for i, string_ in enumerate(f):
            name = "sample_" + str("%10.7o"% i) + '.' + ext
            with open(data_folder + name, 'w') as f:
                f.write(string_)
            counter.update(tokenizer.tokenize(string_))
    return counter
=======
def build_vocab(filepath, data_folder, ext):
    #counter = Counter()
    answer = dict()
    with io.open(filepath, encoding="utf8") as f:
        for i, string_ in enumerate(f):

            name = "sample_" + str("%10.7o"% i) + '.' + ext
            with open(data_folder + name, 'w') as f:
                f.write(string_)
            answer[i] = string_

    return answer

def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()
>>>>>>> vocab and line by line
