from tqdm import tqdm
import re

kazakh_capital_letters = ['Ә', 'Ғ', 'Қ', 'Ң', 'Ө', 'Ұ', 'Ү', 'Һ', 'І', 'Й', 'Ц', 'У', 'К', 'Е', 'Н', 'Г', 'Ш', 'Щ',
                          'З', 'Х', 'Ф', 'Ы', 'В', 'А', 'П', 'Р', 'О', 'Л', 'Д', 'Ж', 'Э', 'Я', 'Ч', 'С', 'М', 'И',
                          'Т', 'Б', 'Ю']
kazakh_common_letters = ['а', 'ә', 'б', 'в', 'г', 'ғ', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'қ', 'л', 'м', 'н', 'ң',
                         'о', 'ө', 'п', 'р', 'с', 'т',  'у', 'ұ', 'ү', 'ф', 'х', 'һ', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'і',
                         'ь', 'э', 'ю', 'я']

filenames_sentences = open('data/sentences_original.txt', 'r', errors='ignore')
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


def splitter(text):
    start = 0
    sentences = []
    for i in re.finditer(f"[\.?!;][\[\]\(\)'\"«"»—]*\s*[0-9\[\]\(\)'\"«»—]*[ЁІАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯҒҚҢҮҰҺӘӨ]",
                         text):
        end = i.start() + 1
        sentences.append(text[start:end].strip())
        start = end
    sentences.append(text[start:].strip())
    sentences = [s for s in sentences if s and s != '\n']
    return sentences


with open(file_sentences_modified, 'w') as fsm: 
    for i, line in tqdm(enumerate(lines)):
        sentences_1 = splitter(line)
        for sentence in sentences_1:
            fsm.write(sentence + '\n')
