import pandas as pd
import numpy as np
from src import preprocessing, config
from sklearn.decomposition import MiniBatchNMF
from sklearn.base import BaseEstimator, TransformerMixin


def get_from_series(y, key):
    return y.loc[key, "concat_topics"] if key in y.index else ""


class CreateAuthorsTopicsFeature(BaseEstimator, TransformerMixin):
    def __init__(self, path_to_encoding=config.path_data_author_encoding):
        self.map_authors = pd.read_csv(path_to_encoding, index_col=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["authors_encoded_by_topic"] = X_.apply(
            lambda x: " ".join((get_from_series(self.map_authors, author) for author in x["authors_split"])), axis=1
        )
        X_["authors_encoded_by_topic"] = X_["authors_encoded_by_topic"].str.replace("  ", " ").str.strip()
        return X_


def build_authors_topics_series(
    df, path=config.path_data_author_encoding, n_topics=12, max_topic_per_author=4, random_state=0
):

    cvSummary = preprocessing.get_tfidf(ngram_range=(1, 2), max_features=1_000, stemming=True)
    Xsummary = cvSummary.fit_transform(df["summary"])

    mbnmf = MiniBatchNMF(
        n_components=n_topics,
        random_state=random_state,
        beta_loss="frobenius",
        alpha_W=0.00005,
        alpha_H=0.00005,
        batch_size=128,
        l1_ratio=0.5,
        init="nndsvda",
    ).fit(Xsummary)

    dict_authors = {}
    dict_authors_topics = {}
    for idx, row in df.iterrows():
        vec_raw = mbnmf.transform(Xsummary[idx, :]).reshape(-1)
        for author in row["authors_split"]:
            dict_authors[author] = dict_authors.get(author, np.zeros(n_topics)) + vec_raw
    for author, vec in dict_authors.items():
        list_topics = np.sort(np.flip(np.argsort(vec))[:max_topic_per_author] + 1)
        dict_authors_topics[author] = list_topics
    y = pd.DataFrame(dict_authors_topics).transpose()
    y["concat_topics"] = y.astype(str).agg(" ".join, axis=1)
    y.to_csv(path)
    return y
