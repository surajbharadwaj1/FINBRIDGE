import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def tokenize(text):
    return word_tokenize(text.lower())

def remove_punctuation(tokens):
    return [t for t in tokens if t not in string.punctuation]

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]

def stem(tokens):
    return [stemmer.stem(t) for t in tokens]

def lemmatize(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text, use_stemming=False):
    tokens = tokenize(text)
    tokens = remove_punctuation(tokens)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem(tokens)
    else:
        tokens = lemmatize(tokens)
    return tokens

def preprocess_to_string(text, use_stemming=False):
    tokens = preprocess(text, use_stemming)
    return " ".join(tokens)