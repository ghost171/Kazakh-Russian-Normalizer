from tqdm import tqdm
import re

kazakh_capital_letters = ['Ә', 'Ғ', 'Қ', 'Ң', 'Ө', 'Ұ', 'Ү', 'Һ', 'І', 'Й', 'Ц', 'У', 'К', 'Е', 'Н', 'Г', 'Ш', 'Щ', 'З', 'Х', 'Ф' , 'Ы', 'В', 'А', 'П', 'Р', 'О', 'Л', 'Д', 'Ж', 'Э', 'Я', 'Ч', 'С', 'М', 'И', 'Т', 'Б', 'Ю']
kazakh_common_letters = ['а', 'ә', 'б', 'в', 'г', 'ғ', 'д','е','ё','ж','з','и','й','к','қ','л','м','н','ң','о','ө','п','р','с','т','у','ұ','ү','ф','х','һ','ц','ч','ш','щ','ъ','ы','і','ь','э','ю','я']

filenames_sentences = open('data/sents_Yerzhan.txt', 'r', errors='ignore')
lines = filenames_sentences.readlines()
file_sentences_modified = 'data/sentences_modified.txt'

def split_point(data, point_of_split):
    result = []
    result.extend(data.split(point_of_split))
    return result

def split_point_list(data, point_of_split):
    result = []
    for elem in data:
        result.extend(elem.split(point_of_split))        
    return result


import re

def splitter(text):
    start = 0
    sents = []
    for i in re.finditer(f"[\.?!;][\[\]\(\)'\"«»—]*\s*[0-9\[\]\(\)'\"«»—]*[ЁІАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯҒҚҢҮҰҺӘӨ]", text):
        next = i.start()+1
        sents.append(text[start:next].strip())
        start = next
    sents.append(text[start:].strip())
    sents = [s for s in sents if s and s != '\n']
    return sents


#print(splitter(example))

'''example_modified = ''
for index, letter in enumerate(example):
    if letter in kazakh_capital_letters and example[index - 1] in kazakh_common_letters:
        example_modified += ". "
        example_modified += letter
        print("ZAHODIT")
    else:
        example_modified += letter

#print(example_modified)

sentences_1 = split_point(example_modified, ". ")
sentences_2 = split_point_list(sentences_1, "! ")
sentences_3 = split_point_list(sentences_2, "? ")

print(sentences_3)'''

'''lines_modified = []
for i, line in tqdm(enumerate(lines)):
    line_modified = ''
    for index, letter in enumerate(line):
        if letter in kazakh_capital_letters and line[index - 1] in kazakh_common_letters:
            line_modified += '. '
            line_modified += letter
        else:
            line_modified += letter
        
    lines_modified.append(line_modified)'''

print("LINES")
print(len(lines))
print("LINES")

with open(file_sentences_modified, 'w') as fsm: 
    for i, line in tqdm(enumerate(lines)):
        sentences_1 = splitter(line)
        for sentence in sentences_1:
            fsm.write(sentence + '\n')
