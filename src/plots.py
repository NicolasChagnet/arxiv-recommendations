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


# Found on the documentation:
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
def plot_top_words(model, feature_names, title, n_top_words=20, nrows_cols=(4, 5)):
    fig, axes = plt.subplots(nrows_cols[0], nrows_cols[1], figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
