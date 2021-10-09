from tqdm import tqdm
import re

def symbols_cleaner(text):
    return re.sub(r'['+chars_unwanted+']', '', text)

def split_point(data, point_of_split):
    result = []
    result.extend(data.split(point_of_split))
    return result

def split_point_list(data, point_of_split):
    result = []
    for elem in data:
        result.extend(elem.split(point_of_split))        
    return result

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

def iterate_through_lines(text):
    return iter(text.splitlines())

kazakh_capital_letters = ['Ә', 'Ғ', 'Қ', 'Ң', 'Ө', 'Ұ', 'Ү', 'Һ', 'І', 'Й', 'Ц', 'У', 'К', 'Е', 'Н', 'Г', 'Ш', 'Щ', 'З', 'Х', 'Ф' , 'Ы', 'В', 'А', 'П', 'Р', 'О', 'Л', 'Д', 'Ж', 'Э', 'Я', 'Ч', 'С', 'М', 'И', 'Т', 'Б', 'Ю']
kazakh_common_letters = ['а', 'ә', 'б', 'в', 'г', 'ғ', 'д','е','ё','ж','з','и','й','к','қ','л','м','н','ң','о','ө','п','р','с','т','у','ұ','ү','ф','х','һ','ц','ч','ш','щ','ъ','ы','і','ь','э','ю','я']

filenames_sentences = open('data/sentences_original.txt', 'r', errors='ignore')
lines = filenames_sentences.readlines()
file_sentences_modified = 'data/sentences_modified.txt'

filenames_sentences_1 = open('data/sentences_original.txt', 'r', errors='ignore')

train_texts_all = filenames_sentences_1.read()

unique_symbols = set(train_texts_all)

kaz_letters_str = set('әіңғүұқөһйцукенгшщзхъфывапролджэячсмитьбюё')
en_letters = set('qwertyuiopasdfghjklzxcvbnm')
numbers = set('0123456789')
special_symbols = set("%-',.*() ")

symbols_to_be_deleted = unique_symbols.difference(set.union(kaz_letters_str, en_letters, numbers, special_symbols))

symbols_to_be_deleted_ = ''.join(list(symbols_to_be_deleted))
print(f'These symbols are to be removed from the dataset: {symbols_to_be_deleted_}') # these symbols are to be deleted from texts

chars_unwanted = re.escape(symbols_to_be_deleted_)

with open(file_sentences_modified, 'w') as fsm: 

    for i, line in tqdm(enumerate(lines)):
        sentence = symbols_cleaner(line)
        sentences_1 = splitter(sentence)
        for sentence in sentences_1:
            fsm.write(sentence + '\n')
