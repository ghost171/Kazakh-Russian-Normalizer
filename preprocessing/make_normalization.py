from num2word.normalizer import number_converter
from tqdm import tqdm

filenames_sentences = open('/home/ghost17/Annotated_4/Kazakh-Russian-Normalizer/Kazakh-Russian-Normalizer/data/sentences_modified.txt', "r", errors='replace')
print("READLINES")
lines = filenames_sentences.readlines()
print("READLINES")

print(len(lines))
for i, line in tqdm(enumerate(lines)):
    filename_unnorm = 'data/raw/unnormalized/sentence_%07d' % (i+1) + '.txt'
    with open(filename_unnorm, 'w', errors='replace') as fn:
        fn.write(line)

    normalized = number_converter(line)

    filename_norm = 'data/raw/normalized/sentence_%07d' % (i+1) + '.txt'
    with open(filename_norm, 'w', errors='replace') as fn:
        fn.write(normalized)

