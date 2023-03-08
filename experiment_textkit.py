import os

dir_path = os.path.dirname(os.path.realpath(__file__))
import copy
import string
from collections import Counter
from collections.abc import Iterable

# from pycontractions import Contractions
import chardet
import pandas as pd
import unidecode
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

stop_words = stopwords.words('english')
porter = PorterStemmer()
## TODO: download the files only during module import, but not module install
# cont = Contractions(os.path.join(dir_path, 'GoogleNews-vectors-negative300.bin.gz'))
tqdm.pandas()


def detect_encoding(data):
    """detect and return the encoding of the data (or list of encoding if the data is enumerable)
     What is the input type?
    """

    result = None

    if isinstance(data, Iterable):

        result = []

        for t in data:
            result.append(chardet.detect(t))

    else:

        result = chardet.detect(data)

    return result


def detect_and_decode(data, default='utf-8'):
    """Detect encoding of data with chardet, and decode accordingly. Return decoded string (or list of decoded string if the data is enumerable)
    what is the input data type?

    """

    result = None

    if isinstance(data, Iterable):
        result = []

        for t in data:
            result.append(chardet.detect(t)['encoding'] or default)

    else:

        result = data.decode(chardet.detect(data)['encoding'] or default)

    return result


def get_encoding_set_from_data(data):
    """Return encoding scheme (or set of encoding scheme if the data is enumerable)
    what is the input data type?

    """

    encoding_set = []

    if not isinstance(data, Iterable):
        return detect_encoding(data)

    for i in data:

        temp = chardet.detect(i)

        if temp['encoding'] in encoding_set:

            continue

        else:

            encoding_set.append(temp['encoding'])

    return sorted(encoding_set)


def decode_accordingly(data, encodings):
    """Decode data with the list of encodings provided at a one-to-one match. Return decoded string (or list of decoded string if the data is enumerable)

    what is the input data type?

    """

    if (not isinstance(encodings, Iterable)):
        return data.decode(encodings)

    if len(data) != len(encodings):
        print('\'data\' and \'encoding\' are not of the same length.')
        return

    if not all(isinstance(n, str) for n in encodings):
        print('\'encodings\' contains non string element(s).')
        return

    return [x.decode(y) for x, y in zip(data, encodings)]


def count_occurrences(text, target):
    """
    what is the input data type?
    """

    if not (isinstance(target, Iterable)):
        target = [target]

    counter = Counter(text)

    return sum([counter[x] for x in target if x in counter])


def count_digits(text):
    """
    what is the input data type?
    """

    return count_occurrences(text, target=string.digits)


def count_alphabets(text):
    """
    what is the input data type?
    """
    return count_occurrences(text, target=string.ascii_letters)


def count_alphabets_upper(text):
    """
    what is the input data type?
    """
    return count_occurrences(text, target=string.ascii_uppercase)


def count_alphabets_lower(text):
    """
    what is the input data type?
    """
    return count_occurrences(text, target=string.ascii_lowercase)


def count_punctuations(text):
    """
    what is the input data type?
    """
    return count_occurrences(text, target=string.punctuation)


def count_whitespaces(text):
    """A string containing all characters that are considered whitespace.
       On most systems this includes the characters space, tab, linefeed,
       return, formfeed, and vertical tab.

    """

    return count_occurrences(text, target=string.whitespace)


def flag_dups(text_series):
    '''Return a series with the duplicates ID 

    '''

    text_series_wo_whitespace = text_series.str.lower().str.replace(' ', '')

    text_series_duplicated = text_series_wo_whitespace[text_series_wo_whitespace.duplicated(keep=False)]

    dups_id_series = text_series_duplicated.groupby(text_series_duplicated).ngroup()

    result_series = pd.Series(data=-1, index=text_series.index)

    result_series = result_series.combine(dups_id_series, max)

    return result_series


def text_process(s, lower_case=True, decode=True, punc=False, num=False, stopWords=False,
                 stemming=True):
    '''Turn a string to processed string, and then to word tokens

    '''
    _s = copy.copy(s)

    # 1. Lower casing

    if lower_case:
        _s = _s.lower()

    # 2. Convert accented characters to English
    if decode:
        _s = unidecode.unidecode(_s)

    # 3. Expand contractions: e.g. I'm --> I am
    # if contractions:
    #     _s = list(cont.expand_texts([_s], precise=True))[0]

    # 4. Remove punctuations (except hyphens)
    if not punc:
        _s = _s.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
        _s = _s.replace('-', ' ')

    # 5. Remove Number
    if not num:
        _s = ''.join(filter(lambda x: not x.isdigit(), _s))

    # 6. Tokenize Text
    tokens = word_tokenize(_s)

    # 7. Remove Stopwords
    if not stopWords:
        tokens = list(filter(lambda x: x not in stop_words, tokens))

    # 8. Stemming
    if stemming:
        tokens = list(map(porter.stem, tokens))

    return tokens


def data_cleaning(df, text_column):
    cleaned_df = df[[text_column]].copy()

    # 1. Identify exact duplicates columns
    cleaned_df['dups_id'] = flag_dups(df[text_column])

    # 2. Process raw text to something the machine learning algorithm can read
    cleaned_df['description_processed'] = cleaned_df[text_column].apply(text_process)

    return cleaned_df
