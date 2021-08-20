import re
import numpy as np  
from num2word.lang_KZ import Num2Word_KZ
from transliterate import translit
import transliterate

#import nltk

kaz_letters_str = '[әіңғүұқөһйцукенгшщзхъфывапролджэячсмитьбюё]'
eng_consonants_str = {'ң','ғ','қ','һ','й','ц','к','н','г','ш','щ','з','х','ъ','ф','в','п','р','л','д','ж','ч','с','м','т','ь','б','b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z','B','C','D','F','G','H','J','K','L','M','N','P','Q','R','S','T','V','W','X','Y','Z'}

converter = Num2Word_KZ()
# zhalgaular_df = pd.read_csv(filepath_zhalgaular, names = ['zhalgaular'])
# zhalgaular = zhalgaular_df.text.to_list()

filepath_zhalgaular = './zhalgaular.txt'
zhalgaular = []

with open (filepath_zhalgaular, "r") as myfile:
    for line in myfile:
        line = line.strip()
        #line = line.lower()
        zhalgaular.append(line)
        
def str2number(number_str):
    #print(number_str)
    if ',' in number_str or '.' in number_str:
        #print('yes')
        number_str = number_str.replace(',', '.')
        int_part, decimal = number_str.split('.')
        cond = True
        for i in decimal:
            if i != '0':
                cond = False
        if not cond:
            return float(number_str)
        else:
            return int(int_part)
    else:
        return int(number_str)

def is_consonant(ch):

    if(ch=='A' or ch=='a' or ch=='E' or ch =='e' or ch=='I' or ch=='i' or ch=='O' or ch=='o' or ch=='U' or ch=='u'):
        return False
    else:
        return True
    
remarked = 0


def romanToInt(s):
    roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900}
    i = 0
    num = 0
    while i < len(s):
        if i+1<len(s) and s[i:i+2] in roman:
            num+=roman[s[i:i+2]]
            i+=2
        else:
            num+=roman[s[i]]
            i+=1
    return num

def converter_1(text):
    # to normalize any decimal number regardless, whether it is followed or not by hyphen symbol
    number_fragment_list = re.findall(r'([0-9]+\,[0-9]+|[0-9]+\.[0-9]+|\-[0-9]+\.[0-9]+|\-[0-9]+\,[0-9]+)', text, re.U)
    #print(number_fragment_list)
    remarked = 0
    if len(number_fragment_list) != 0:
        remarked = 1
        for number_fragment in number_fragment_list:
            #if number_fragment.endswith('-'):
            #    converted_number = converter.to_cardinal(str2number(temp_item[:-1]))
            #else:
            #    converted_number = converter.to_cardinal(str2number(temp_item))
            #text = re.sub(temp_item, converted_number, text)
            text = text.replace(number_fragment, converter.to_cardinal(str2number(number_fragment)), 1)
        return text, remarked
    else:
        return text, remarked

def converter_2(text):
    # to normalize any int number followed by hyphen symbol and text by correctly identying the ordinal/cardinal form
    number_word_fragment_list = re.findall(r'\b[0-9]+-\b' + kaz_letters_str + '+', text, re.U)
    #print(number_word_fragment_list)
    remarked = 0
    if len(number_word_fragment_list) != 0:
        remarked = 1
        for number_word_fragment in number_word_fragment_list:
            word_fragment = re.sub('[0-9]+-', '', number_word_fragment)
            number_fragment = re.sub('-\w+', '', number_word_fragment)
            #print(word_fragment)
            if any(zhalgau == word_fragment for zhalgau in zhalgaular):
                converted_number = converter.to_cardinal(str2number(number_fragment))
                text = re.sub(number_word_fragment, converted_number + word_fragment, text)
                #break
                #return text
            
            else:
                converted_number = converter.to_ordinal(str2number(number_fragment))
                text = re.sub(number_word_fragment, converted_number + ' ' + word_fragment, text)
                #break
                #return text
        return text, remarked
    else:
        return text, remarked

def converter_3(text):
    # to normalize interval of numbers specified by hyphen symbol by correctly identying the ordinal/cardinal form
    keywords = ['ғасыр', 'жыл', 'қаңтар', 'ақпан', 'наурыз', 'сәуір', 'мамыр', 'маусым', 'шілде', 'тамыз', 'қыркүйек', 'қазан', 'қараша', 'желтоқсан']
    months = ['ғасыр', 'қаңтар', 'ақпан', 'наурыз', 'сәуір', 'мамыр', 'маусым', 'шілде', 'тамыз', 'қыркүйек', 'қазан', 'қараша', 'желтоқсан']
    number_word_fragment_list = re.findall(r'[0-9]+\-[0-9]+\s\w+', text, re.U)
    #print(number_word_fragment_list)
    remarked = 0
    if len(number_word_fragment_list) != 0:
        remarked = 1
        for number_word_fragment in number_word_fragment_list:
            word_fragment = re.sub('[0-9]+\-[0-9]+\s', '', number_word_fragment)
            number_fragment = re.sub('\s\w+', '', number_word_fragment)
            number_1, number_2 = number_fragment.split('-')
            to_ordinal = False
            
            keyword_found = next((item for item in keywords if word_fragment.startswith(item)), None)
            
            if keyword_found == 'жыл' and int(number_1) > 999 and int(number_2) > 999:
                remaining = re.sub(keyword_found, '', word_fragment)
                if remaining in zhalgaular or remaining == '':
                    to_ordinal = True
          
            if keyword_found in months and int(number_1) > 0 and int(number_2) < 32:
                
                remaining = re.sub(keyword_found, '', word_fragment)
                if remaining in zhalgaular or remaining == '':
                    to_ordinal = True
            if to_ordinal:
                number_word_fragment_sub = converter.to_ordinal(int(number_1)) + '-' + converter.to_ordinal(int(number_2)) + ' ' + word_fragment
                text = text.replace(number_word_fragment, number_word_fragment_sub, 1)
            else:
                number_word_fragment_sub = converter.to_cardinal(int(number_1)) + '-' + converter.to_cardinal(int(number_2)) + ' ' + word_fragment
                text = text.replace(number_word_fragment, number_word_fragment_sub, 1)
        return text, remarked
    else:
        return text, remarked

def converter_4(text):
    # to normalize digital times specified by :
    number_fragment_list = re.findall(r'[0-9]+:[0-9]+', text, re.U)
    #print(number_fragment_list)
    remarked = 0
    if len(number_fragment_list) != 0:
        remarked = 1
        for number_fragment in number_fragment_list:
            hour, minute = number_fragment.split(':')
            if minute.startswith('0') and minute != '0':
                    minute_1, minute_2 = minute[0], minute[1]
                    number_fragment_sub = ' '.join([converter.to_cardinal((int(hour))), converter.to_cardinal(int(minute_1)), converter.to_cardinal(int(minute_2))])

            else:
                number_fragment_sub = ' '.join([converter.to_cardinal((int(hour))), converter.to_cardinal(int(minute))])
            text = re.sub(number_fragment, number_fragment_sub, text)
        return text, remarked 
    else:
        return text, remarked

def converter_5(text):
    # to normalize numbers followed by keywords by correctly identying the ordinal/cardinal form
    keywords = ['ғасыр','жыл', 'қаңтар', 'ақпан', 'наурыз', 'сәуір', 'мамыр', 'маусым', 'шілде', 'тамыз', 'қыркүйек', 'қазан',
                'қараша', 'желтоқсан']
    months = ['ғасыр', 'қаңтар', 'ақпан', 'наурыз', 'сәуір', 'мамыр', 'маусым', 'шілде', 'тамыз', 'қыркүйек', 'қазан', 'қараша',
              'желтоқсан']
    number_word_fragment_list = re.findall(r'[0-9]+\s' + kaz_letters_str + '+', text, re.U)
    # print(number_word_fragment_list)
    remarked = 0 
    if len(number_word_fragment_list) != 0:
        remarked = 1
        for number_word_fragment in number_word_fragment_list:
            word_fragment = re.sub('[0-9]+\s', '', number_word_fragment)
            number_fragment = re.sub('\s\w+', '', number_word_fragment)
            to_ordinal = False
            keyword_found = next((item for item in keywords if word_fragment.startswith(item)), None)

            if keyword_found == 'жыл' and int(number_fragment) > 999:
                remaining = re.sub(keyword_found, '', word_fragment)
                if remaining in zhalgaular or remaining == '':
                    to_ordinal = True

            if keyword_found in months and int(number_fragment) > 0 and int(number_fragment) < 32:

                remaining = re.sub(keyword_found, '', word_fragment)
                if remaining in zhalgaular or remaining == '':
                    to_ordinal = True
            if to_ordinal:
                number_word_fragment_sub = converter.to_ordinal(int(number_fragment)) + ' ' + word_fragment
                text = re.sub(number_word_fragment, number_word_fragment_sub, text)
            else:
                number_word_fragment_sub = converter.to_cardinal(int(number_fragment)) + ' ' + word_fragment
                text = re.sub(number_word_fragment, number_word_fragment_sub, text)
        return text, remarked
    else:
        return text, remarked

def converter_6(text):
    # to normalize any remaining number(s)
    number_fragment_list = re.findall(r'[0-9]+', text, re.U)
    #print(number_fragment_list)
    remarked = 0 
    if len(number_fragment_list) != 0:
        remarked = 1
        for number_fragment in number_fragment_list:
            number_fragment_sub = converter.to_cardinal(int(number_fragment))
            #print(number_fragment, number_fragment_sub)
            #print(text)
            text = text.replace(number_fragment, number_fragment_sub, 1)
        return text, remarked
    else:
        return text, remarked

def converter_7(text):
    # to normalize digital times specified by :
    number_fragment_list = re.findall(r'\b[0-9]:[0-9]\b', text, re.U)
    #print(number_fragment_list)
    remarked = 0
    if len(number_fragment_list) != 0:
        remarked = 1
        for number_fragment in number_fragment_list:
            number_1, number_2 = number_fragment.split(':')
            number_fragment_sub = ' '.join([converter.to_cardinal((int(number_1))), converter.to_cardinal(int(number_2))])
            text = re.sub(number_fragment, number_fragment_sub, text)
        return text, remarked
    else:
        return text, remarked



def converter_8(text):
    # \d-\d-\w+
    number_word_fragments = re.findall(r'(\b[0-9]+-[0-9]+-' + kaz_letters_str + ')', text, re.U) 
    remarked = 0
    if len(number_word_fragments) != 0:
        remarked = 1
        for number_word_fragment in number_word_fragments:
            word_fragment = re.sub('[0-9]+-[0-9]+-', '', number_word_fragment)
            number_fragment = re.sub('-' + word_fragment, '', number_word_fragment)
            to_ordinal = False
            if any(zhalgau == word_fragment for zhalgau in zhalgaular):
                number_1, number_2 = number_fragment.split('-')
                converted_number_1, converted_number_2 = converter.to_cardinal(int(number_1)), converter.to_cardinal(int(number_2))
                text = text.replace(number_word_fragment, converted_number_1 + '-' + converted_number_2 + word_fragment)
            
            else:
                number_1, number_2 = number_fragment.split('-')
                converted_number_1, converted_number_2 = converter.to_ordinal(int(number_1)), converter.to_ordinal(int(number_2))
                text = text.replace(number_word_fragment, converted_number_1 + '-' + converted_number_2 + word_fragment)
        return text, remarked
    else:
        return text, remarked
    

def converter_roman(text):
    english_roman_numbers = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',\
     'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI',\
     'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI', 'XXXII', 'XXXIII', 'XXXIV', 'XXXV', 'XXXVI', 'XXXVII',\
     'XXXVIII', 'XXXIX']
    kazakh_roman_numbers = ['І', 'ІІ', 'ІІІ', 'ІХ', 'Х', 'ХІ', 'ХІІ', 'ХІІІ', 'ХІХ', 'ХХ', 'ХХІ', 'ХХІІ', \
                            'ХХІІІ', 'ХХІХ', 'ХХХ', 'ХХХІ', 'ХХХІІ', 'ХХХІІІ', 'ХХХІХ']
    roman_numbers = english_roman_numbers + kazakh_roman_numbers
    #words = set(text.split(' '))
    #words = set(nltk.word_tokenize(text))
    words = set(re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\w\-]+", text))
    #print(words)
    roman_numbers_found = words.intersection(set(roman_numbers))
    #number_roman = next((item for item in roman_numbers if item in text), None)
    #print(roman_numbers_found)
    for roman_number_found in roman_numbers_found:
        number_roman_en = re.sub('Х', 'X', roman_number_found)
        number_roman_en = re.sub('І', 'I', number_roman_en)
        arabic_number = romanToInt(number_roman_en)
        converted_number = converter.to_ordinal(arabic_number)
        text = text.replace(roman_number_found, converted_number, 1)
    return text

def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]

def converter_for_minus(text):
    for minus in re.finditer(r'-[0-9]+', text):
        index = minus.start()
        if(text[index - 1].isdigit()):
            text = replacer(text, " минус ", index)
    return text

def plus_converter(text):
    for minus in re.finditer(r'[+]+', text):
        index = minus.start()

        text = replacer(text, " плюс ", index)
    return text


def multiply_converter(text):
    for minus in re.finditer(r'[*]+|[x]+', text):
        index = minus.start()
        text = replacer(text, " көбейту ", index)
    return text

def devide_converter(text): 
    for minus in re.finditer(r'[\/]+', text):
        index = minus.start()
        text = replacer(text, " бөлу ", index)
        
    return text


def english_converter(text):

    word_fragment_list = re.findall(r'[a-zA-Z0-9]+', text, re.U)
    #print(word_fragment_list)

    for i, letter in enumerate(text):
        if letter == 'w' and (is_consonant(text[i + 1]) or is_consonant(text[i - 1])):
            text = text[:i] + 'в' + text[i+1:]
        if letter == 'W' and (text[i + 1] in eng_consonants_str or text[i - 1] in eng_consonants_str):
            #print("KEK")
            text = text[:i] + 'В' + text[i+1:]

    for word_fragment in word_fragment_list:
        word_fragment_converted = transliterate.translit(word_fragment, 'ru')
        text = text.replace(word_fragment, word_fragment_converted)

    #text = text.replace('w', 'у')
    #text = text.replace('W', 'У')
    #text = text.replace('Q', 'К')
    #text = text.replace('q', 'к')
    #text = text.replace('X', 'КС')
    #text = text.replace('x', 'кс')
    #text = text.replace('B', 'Б')
    #text = text.replace('b', 'б')

    #for i, letter in enumerate(text):
        #if letter == 'c' and (text[i + 1] == 'O' or text[i + 1] == 'о'):
        #    text = text[:i] + 'к' + text[i+1:]
        #if letter == 'С' and (text[i + 1] == 'O' or text[i + 1] == 'о'):
        #    text = text[:i] + 'К' + text[i+1:]
    
    return text

def ampersand_converter(text):
    text = text.replace('&', 'және')
    return text

def number_mark_converter(text):
    text = text.replace('№', 'номер')
    return text

def values_converter_1(text):
    word_fragment_list = re.findall(r'\$[0-9]+', text, re.U)

    for word_fragment in word_fragment_list:
        number_fragment = word_fragment.split('$')
        if (number_fragment[1] == '1'):
            number_fragment_sub = number_fragment[1] + " доллар "
        else:
            number_fragment_sub = number_fragment[1] + " долларлар "

        text = text.replace(word_fragment, number_fragment_sub)
    return text

def values_converter_2(text):
    text = text.replace('$', ' долларлар ')
    return text

def procent_converter_1(text):
    text = text.replace('%-тен', ' проценттен ')
    return text

def procent_converter_2(text):
    text = text.replace('%-ке', ' процентке ')
    return text

def procent_converter_3(text):
    text = text.replace('%', ' проценттер ')
    return text

def defis_substitution(text):
    text = text.replace('-', ' ')
    return text

def butin_mark(text):
    text = text.replace('.', ' ')
    return text

def converter_9(text):
    letter_word_fragments = re.findall(r'' + kaz_letters_str + '-' + kaz_letters_str, text, re.U) 
    for letter_word in letter_word_fragments:
        letter_word_list = letter_word.split('-')
        letter_word_fragment_sub = ''
        for word in letter_word_list:
            letter_word_fragment_sub += ' ' + word
        text = re.sub(letter_word, letter_word_fragment_sub, text)

    return text

def converter_defis_long(text):
    text = text.replace('–', ' ')
    return text

def reductions_of_numbers(text):
    reductions_of_numbers = ['млн', 'млрд']
    text = text.replace('млн', 'миллион')
    text = text.replace('млрд', 'миллиард')

    return text

def number_converter(text):
    #text = converter_9(text)
    #text = values_converter_1(text)
    #text = values_converter_2(text)
    #text = procent_converter_1(text)
    #text = procent_converter_2(text)
    #text = procent_converter_3(text)
    text, remarked_1 = converter_1(text)
    #print("1", text)
    text, remarked_2 = converter_8(text)
    #print("2", text)
    text, remarked_3 = converter_2(text)
    #print("3", text)
    text, remarked_4 = converter_3(text)
    #print("4", text)
    text, remarked_5 = converter_7(text)
    #print("5", text)
    text, remarked_6 = converter_4(text)
    #print("6", text)
    text, remarked_7 = converter_5(text)
    #print("7", text)
    text, remarked_8 = converter_6(text)
    #print("8", text)

    #text = converter_roman(text)

    remarked = 0
    if (remarked_1 or remarked_2 or remarked_3 or remarked_4 or remarked_5 or remarked_6 or remarked_7 or remarked_8):
        remarked = 1

    #print("9", text)
    #text = plus_converter(text)
    #text = multiply_converter(text)
    #text = devide_converter(text)
    #text = english_converter(text)
    #text = ampersand_converter(text)
    #text = number_mark_converter(text)
    #text = butin_mark(text)
    #text = defis_substitution(text)
    #text = converter_defis_long(text)
    #text = reductions_of_numbers(text)
    #text = converter_for_minus(text)
    
    return text, remarked
