import pandas as pd
import numpy as np
import re
import string

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from src import arxiv_utils


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
    """Given a list of author, replaces spaces with underscore and semi-colon with space

    Args:
        title (str): Author list of an arXiv article

    Returns:
        str: Standardized author list
    """
    author_list = authors.split(";")
    author_list = [author.strip().replace(" ", "_") for author in author_list]
    authors = " ".join(author_list)
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
def get_cv():
    return CountVectorizer(stop_words="english", analyzer="word", strip_accents="ascii", token_pattern=r"\b([^\s]+)\b")


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


cvMainCategory = CustomLabelEncoder()

pipeline_arxiv_articles = Pipeline(
    steps=[
        ("Cleaning...", CleanFields()),
        (
            "Feature engineering...",
            BuildWordSoup(["title", "authors"]),
        ),
    ]
)
