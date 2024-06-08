import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_top_n(vocab_1, vocab_2, cols, ntop=10, ax=None):
    """Histograme comparison of top word frequencies between two vocabulary sets.

    Args:
        vocab_1 (pandas.Series): Vocabulary series.
        vocab_2 (pandas.Series): Vocabulary series.
        cols (list(str)): Column names.
        ntop (int, optional): Number of top elements to consider. Defaults to 10.
        ax (optional): Matplotlib axes instance. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    top_1 = set(vocab_1.head(ntop).index)
    top_2 = set(vocab_2.head(ntop).index)
    top_12 = top_1.union(set.intersection(set(vocab_1.index), top_2))
    top_21 = top_2.union(set.intersection(top_1, set(vocab_2.index)))
    n1 = vocab_1.sum()
    n2 = vocab_2.sum()
    top_title_df = (
        pd.DataFrame([vocab_1[list(top_12)] / n1, vocab_2[list(top_21)] / n2])
        .transpose()
        .sort_values(by=0)
        .rename(columns={0: cols[0], 1: cols[1]})
    )
    top_title_df.plot(kind="bar", ax=ax)
