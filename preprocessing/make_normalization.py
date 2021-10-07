import os
from num2word.normalizer import number_converter
from tqdm import tqdm

filenames_sentences = open('data/sentences_modified.txt', "r", errors='replace')
lines = filenames_sentences.readlines()

print('Number of sentences is', len(lines))

export_paths = ['data/raw/unnormalized', 'data/raw/normalized']
for export_path in export_paths:
    if not os.path.exists(export_path):
        os.makedirs(export_path)

for i, line in tqdm(enumerate(lines)):
    filename_unnorm = f'{export_paths[0]}/sentence_%07d' % (i+1) + '.txt'
    with open(filename_unnorm, 'w', errors='replace') as fn:
        fn.write(line)

    normalized = number_converter(line)

    filename_norm = f'{export_paths[1]}/sentence_%07d' % (i+1) + '.txt'
    with open(filename_norm, 'w', errors='replace') as fn:
        fn.write(normalized)

