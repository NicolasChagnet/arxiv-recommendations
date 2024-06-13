import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


def build_similarity_matrix(X, cv):
    """Given a series or dataframe and a transformer to vectorized data, returns the cosine similarity matrix.

    Args:
        X: Series or dataframe containing the text to be tokenized.
        cv: ColumnTransformer instance used to tokenize the text.

    Returns:
        (numpy.ndarray, numpy.ndarray): Cosine similarity matrix and encoded features.
    """
    df_encoded = cv.fit_transform(X)
    sim_mat = cosine_similarity(df_encoded, df_encoded)
    return sim_mat, df_encoded


def build_similarity_line(X, entry_id, cv):
    """Given a series or dataframe and a transformer to vectorized data, returns the cosine similarity matrix.

    Args:
        X: Series or dataframe containing the text to be tokenized.
        entry_id (int): Index of the entry.
        cv: ColumnTransformer instance used to tokenize the text.

    Returns:
        (numpy.ndarray, numpy.ndarray): Cosine similarity scores w.r.t. the entry and encoded features.
    """
    df_encoded = cv.fit_transform(X)
    sim_mat = cosine_similarity(df_encoded[entry_id, :], df_encoded).reshape(-1)
    return sim_mat, df_encoded


def get_similar_entries_line(
    database,
    entry_id,
    similarity_line,
    n_recommendations=10,
):
    """Given an entry and a list of similarity scores, finds recommended entries.

    Args:
        database (pandas.DataFrame): DataFrame containg the original entries' information.
        entry_id (int): Index of base entry.
        similarity_line (numpy.ndarray): List of similarity scores associated to the line.
        weights (list, optional): List of weights for each similarity matrix. Defaults to None (assigned to equal weights).
        n_recommendations (int, optional): Number of recommendations. Defaults to 10.

    Returns:
        pandas.DataFrame: Catalogue of similar entries.
    """
    # Initialization of weights
    # Find top n matches except the element itself
    top_n_idx = np.flip(np.argsort(similarity_line))
    top_n_idx = top_n_idx[top_n_idx != entry_id][0:n_recommendations]
    # Extract their score
    top_n_sim_values = similarity_line[top_n_idx]
    # Only keep those with finite similarity
    top_n_idx = top_n_idx[top_n_sim_values > 0]
    scores = top_n_sim_values[top_n_sim_values > 0]
    # Build resulting output
    res = pd.DataFrame(
        {
            "paper_id": database["id"].iloc[entry_id],
            "paper_title": database["title"].iloc[entry_id],
            "paper_authors": database["authors"].iloc[entry_id],
            "sim_entries_title": database["title"].iloc[top_n_idx].values,
            "sim_entries_authors": database["authors"].iloc[top_n_idx].values,
            "sim_entries_categories": database["categories"].iloc[top_n_idx].values,
            "sim_id": database["id"].iloc[top_n_idx].values,
            "scores": scores,
        },
        columns=[
            "paper_id",
            "paper_title",
            "paper_authors",
            "sim_entries_title",
            "sim_entries_authors",
            "sim_entries_categories",
            "sim_id",
            "scores",
        ],
    )
    return res


def get_similar_entries_entry(
    database,
    entry_id,
    similarity_matrices,
    weights=None,
    n_recommendations=10,
):
    """Given an entry and a similarity matrix, finds recommended entries.

    Args:
        database (pandas.DataFrame): DataFrame containg the original entries' information.
        entry_id (int): Index of base entry.
        similarity_matrices (numpy.ndarray): List of similarity matrices associated to the entries.
        weights (list, optional): List of weights for each similarity matrix. Defaults to None (assigned to equal weights).
        n_recommendations (int, optional): Number of recommendations. Defaults to 10.

    Returns:
        pandas.DataFrame: Catalogue of similar entries.
    """
    # Initialization of weights
    if len(similarity_matrices) == 0:
        raise ValueError("You must supply at least one similarity matrix!")
    similarity_matrices = np.array(similarity_matrices)
    if weights is None or len(weights) != similarity_matrices.shape[0]:
        weights = np.ones(similarity_matrices.shape[0]) / similarity_matrices.shape[0]
    else:
        weights = np.array(weights)
    # Average similarity matrix
    average_sim_mat = (weights * similarity_matrices.transpose()).sum(axis=-1)
    return get_similar_entries_line(
        database, entry_id, average_sim_mat[entry_id, :], n_recommendations=n_recommendations
    )


def average_similarity_library(df, id, sim_mat, top=20, ref="is_in_library"):
    """Evaluates the average cosine similarity of an element with its top recommended neighbors in the library.

    Args:
        df (pandas.DataFrame): DataFrame with all the information.
        id (int): Index of the entry to evaluate.
        sim_mat (numpy.ndarray): Cosine similarity matrix between elements of the dataframe.
        top (int, optional): Number of top recommended papers to consider. Defaults to 20.
        ref (str, optional): Reference column to consider when looking at neighbors. Defaults to "is_in_library".

    Returns:
        float: Average cosine similarity.
    """
    top_n_idx = np.flip(np.argsort(sim_mat[id, :]))
    top_n_idx = top_n_idx[0:top]
    are_in_library = df.loc[top_n_idx, ref]
    top_n_idx_in_library = are_in_library.loc[are_in_library > 0].index
    if top_n_idx_in_library.empty:
        return 0.0
    return sim_mat[id, top_n_idx_in_library].sum() / top
    # return top_n_idx_in_library.shape[0] / top
