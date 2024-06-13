import pandas as pd
import numpy as np
import re
import string

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src import arxiv_utils, recommender, config
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


# Build custom analyzer
stemmer = SnowballStemmer(language="english")


def stem_english_words(tokens):
    return (stemmer.stem(w) for w in tokens)


stop_words_english = stopwords.words("english")
stop_words_english_stemmed = list(stem_english_words(stop_words_english))


class StemmedCountVectorizer(CountVectorizer):
    """Custom CountVectorizer with added stemming."""

    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(stem_english_words(tokenize(doc)))


class StemmedTfidfVectorizer(TfidfVectorizer):
    """Custom TfidfVectorizer with added stemming."""

    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(stem_english_words(tokenize(doc)))


def load_dataset():
    """Loads the dataset and apply the basic pipeline.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    data_raw = pd.read_csv(config.path_data_merged)
    data_clean = pipeline_arxiv_articles.fit_transform(data_raw)
    return data_clean


def clean_text_string(title):
    """Given a string, cleans it by removing extra whitespaces and latex characters

    Args:
        title (str): Title of an arXiv article

    Returns:
        str: Cleaned title
    """
    # First replace specific punctuation
    title = title.replace("-", " ")
    # Remove LaTeX commands
    title = re.sub(r"\\[a-zA-Z]+{([^}]*)}", "\1", title)
    title = re.sub(r"[_\^]{([^}]*)}", "\1", title)
    # Remove extra whitespaces
    title = title.strip()
    # Remove all leftover punctuation
    translator = str.maketrans("", "", string.punctuation)
    title = title.translate(translator)
    return title


def clean_author_list(authors):
    """Given a list of authors, replaces spaces with underscores and semi-colons with spaces.

    Args:
        authors (str): Author list of an arXiv article

    Returns:
        str: Standardized author list
    """
    author_list = authors.split(";")
    author_list = [author.strip().replace(" ", "_") for author in author_list]
    authors = " ".join(author_list)
    return authors


def undo_clean_author_list(authors):
    """Given a standardized list of authors, replaces underscores with spaces and spaces with semi-colons.

    Args:
        authors (str): Standardized author list

    Returns:
        str: Author list of an arXiv article
    """
    author_list = authors.split(" ")
    author_list = [author.strip().replace("_", " ") for author in author_list]
    authors = "; ".join(author_list)
    return authors


class CleanFields(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["title"] = X_["title"].apply(lambda x: clean_text_string(x))
        X_["authors"] = X_["authors"].apply(lambda x: clean_author_list(x))
        X_["authors_split"] = X_["authors"].str.split(" ")
        X_["summary"] = X_["summary"].apply(lambda x: clean_text_string(x))

        return X_


class BuildWordSoup(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["word_soup_" + "_".join(self.cols)] = X_.apply(lambda x: " ".join([x[col] for col in self.cols]), axis=1)
        return X_


# Getter functions to quickly instantiate some CountVectorizers
def get_cv(ngram_range=(1, 1), max_features=None, stemming=False):
    """Wrapper for the CountVectorizer with customized defaults

    Args:
        ngram_range ((int, int), optional): Range of n-grams tokens to consider. Defaults to (1, 1).
        max_features (int, optional): Restriction of vocabulary to top-n maximal features. Defaults to None.
        stemming (bool): Whether to include stemming in the CountVectorizer. Defaults to False.
    Returns:
        CountVectorizer instance with customized initialization
    """
    vectorizer = StemmedCountVectorizer if stemming else CountVectorizer
    stop_words = stop_words_english_stemmed if stemming else stop_words_english
    return vectorizer(
        stop_words=stop_words,
        analyzer="word",
        strip_accents="ascii",
        token_pattern=r"\b([^\s]+)\b",
        ngram_range=ngram_range,
        max_features=max_features,
    )


# Getter functions to quickly instantiate some TfidfVectorizer
def get_tfidf(ngram_range=(1, 1), max_features=None, stemming=False):
    """Wrapper for the TfIdf with customized defaults

    Args:
        ngram_range ((int, int), optional): Range of n-grams tokens to consider. Defaults to (1, 1).
        max_features (int, optional): Restriction of vocabulary to top-n maximal features. Defaults to None.
        stemming (bool): Whether to include stemming in the CountVectorizer. Defaults to False.
    Returns:
        CountVectorizer instance with customized initialization
    """
    vectorizer = StemmedTfidfVectorizer if stemming else TfidfVectorizer
    stop_words = stop_words_english_stemmed if stemming else stop_words_english

    return vectorizer(
        stop_words=stop_words,
        analyzer="word",
        strip_accents="ascii",
        token_pattern=r"\b([^\s]+)\b",
        ngram_range=ngram_range,
        max_features=max_features,
    )


# For the main category, we use a custom label encoder.
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(arxiv_utils.get_arxiv_categories())
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["primary_category_encoded"] = self.encoder.transform(X["primary_category"])
        return X_


class BuildAverageSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, cos_sim_mat, top=20):
        self.cos_sim_mat = cos_sim_mat
        self.top = top

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["average_similarity_library"] = 0.0
        for idx, _ in X_.iterrows():
            X_.loc[idx, "average_similarity_library"] = recommender.average_similarity_library(
                X_,
                idx,
                self.cos_sim_mat,
                top=self.top,
            )
        X_["average_similarity_library_normalized"] = (
            MinMaxScaler().fit_transform(X_[["average_similarity_library"]]).reshape(-1)
        )
        X_["above_theshold"] = X_["average_similarity_library_normalized"] > config.threshold_similarity_normalized
        X_["target"] = X_["above_theshold"] | X_["is_in_library"]
        return X_


cvMainCategory = CustomLabelEncoder()
cvMixed = ColumnTransformer(
    transformers=[
        ("title", get_cv(ngram_range=(1, 2), max_features=10_000, stemming=True), "title"),
        ("authors", get_cv(ngram_range=(1, 1), max_features=1_000), "authors"),
        ("categories", get_cv(ngram_range=(1, 1), max_features=20), "categories"),
        (
            "summary",
            get_tfidf(ngram_range=(1, 2), stemming=True),
            "summary",
        ),
    ],
    remainder="drop",
)

cvMixedEncoding = ColumnTransformer(
    transformers=[
        ("title", get_cv(ngram_range=(1, 2), max_features=10_000, stemming=True), "title"),
        ("authors_encoded_by_topic", get_tfidf(ngram_range=(1, 1)), "authors_encoded_by_topic"),
        # ("categories", get_cv(ngram_range=(1, 1), max_features=20), "categories"),
        (
            "summary",
            get_tfidf(ngram_range=(1, 2), stemming=True),
            "summary",
        ),
    ],
    remainder="drop",
)

pipeline_arxiv_articles = Pipeline(
    steps=[
        ("Cleaning...", CleanFields()),
    ]
)
