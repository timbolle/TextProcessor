from sklearn.datasets import fetch_20newsgroups
from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition

import nltk

nltk.data.path.append(r"D:\Anaconda\envs\Giulia\Lib\site-packages\nltk\nltk_data")

print("TextPreprocessor : v1.2")


class SourceCodeCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, css_selector="h1, h2, p"):
        return [self.sc_cleaner(doc, css_selector) for doc in X]

    @staticmethod
    def sc_cleaner(document, css_selector="h1, h2, p"):
        if not (isinstance(document, str) or isinstance(document, bytes)):
            return None
        soup = BeautifulSoup(document, 'html.parser')
        text = lambda x: x.text

        elem = map(text, soup.select(css_selector))

        output = []
        for e in elem:
            if e not in output and e not in [None, ""]:
                output.append(e)

        return "\n".join(output)


class NLTK_Tokenizer(BaseEstimator, TransformerMixin):
    """
    NLTK tokenizer : split text into sentences and sentences into words
    Can return a list of all the tokens in the document
    """

    def fit(self, X, y=None):
        """
        Return itself because no need to fit
        :param X: a list or iterable of raw strings, each representing a document.
        :param y: a list or iterable of labels, which will be label encoded.
        :return: self
        """
        return self

    def transform(self, X):
        """
        Tokenize each document, creating a list of token
        :param X: a list or iterable of raw strings, each representing a document.
        :return: A list of token for each documents
        """
        return [list(self.tokenize(doc)) for doc in X]

    @staticmethod
    def tokenize(document):
        for sentence in sent_tokenize(document):
            for token in wordpunct_tokenize(sentence):
                yield token


class NLTK_Cleaner(BaseEstimator, TransformerMixin):
    """
    NLTK cleaner: remove stopword and punctuation
    Need to be tokenized
    """

    def __init__(self, stopwords=None, punct=None):
        """
        Init the cleaner
        :param stopwords: list of stopwords, if not set: nltk.corpus.stopwords.words('english')
        :param punct: list of punctuation, if not set: string.punctuation
        """

        self.stopwords = stopwords if stopwords else set(sw.words('english'))
        self.punct = punct if punct else set(string.punctuation)

    def fit(self, X, y=None):
        """
        Return itself because no need to fit
        :param X: a list or iterable of raw strings, each representing a document.
        :param y: a list or iterable of labels, which will be label encoded.
        :return: self
        """
        return self

    def transform(self, X):
        """
        Clean each document, as a list of token
        :param X: a list or iterable of token list, each representing a document.
        :return: A list of token for each documents
        """
        return [list(self.cleaner(doc)) for doc in X]

    def cleaner(self, document):
        """
        Clean the document (lower> strip > stopwords > punctuation)
        :param document: a list of token
        :return: generator with the cleaned token
        """
        for token in document:
            token = token.lower()
            token = token.strip()
            if token in set(self.stopwords) or all(char in set(self.punct) for char in token):
                continue

            yield token


class NLTK_Lemmatizer(BaseEstimator, TransformerMixin):
    """
    Lemmatize the token in each document
    """

    def __init__(self, lemmatizer=None):
        self.lemmatizer = lemmatizer if lemmatizer else WordNetLemmatizer()

    def fit(self, X, y=None):
        """
        Return itself because no need to fit
        :param X: a list or iterable of raw strings, each representing a document.
        :param y: a list or iterable of labels, which will be label encoded.
        :return: self
        """
        return self

    def transform(self, X):
        """
        Lemmatize each token in the document collection
        :param X: a list or iterable of token list, each representing a document.
        :return: A list of token for each documents
        """
        return [list(self.lemmatize(doc)) for doc in X]

    def lemmatize(self, document):
        """
        lemmatize a document, using pos_tag from nltk
        :param document: a list a token
        :return: generator with the lemmas
        """
        for token, tag in pos_tag(document):
            lemma = self.lemmatize_one(token, tag)
            yield lemma

    def lemmatize_one(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


if __name__ == '__main__':
    data_train = fetch_20newsgroups(subset='train', shuffle=True)
    print(data_train.target_names)

    tokenized = NLTK_Tokenizer().transform(data_train.data)
    cleaned = NLTK_Cleaner(stopwords= ["un","il"]).transform(tokenized)
    lem = NLTK_Lemmatizer().transform(cleaned)

    tfidf_transformer = TfidfVectorizer(tokenizer=lambda x: x,
                                        lowercase=False)  # also possible to countVector et Tfifdf transformer
    model = tfidf_transformer.fit(lem)
    data = model.transform(lem)

    pca = decomposition.PCA(n_components=2)
    pca.fit(data)

    data_red = pca.transform(data)
