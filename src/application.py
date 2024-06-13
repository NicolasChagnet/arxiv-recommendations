import logging
import pickle
import pandas as pd
from src import arxiv_utils, config, preprocessing, recommender, feature_engineering

logger = logging.getLogger("arxiv_recommender")

# Log to file
fh = logging.FileHandler(config.path_log)
fh.setLevel(logging.INFO)
logger.addHandler(fh)


def find_recommendations(arxiv_id, n_recommendations=15, encode_authors_topic=False):
    """Given an arXiv identifier, prints and returns top recommendations from within the local database.

    Args:
        arxiv_id (str): arXiv identifier of the form "2308.1508" or "hep-th/9802150".
        n_recommendations (int, optional): Number of recommendations to return. Defaults to 15.
        encode_authors_topic (bool, optional): Whether to encode authors by their most published topic in the database. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame of recommendations
    """
    # Loading article
    logger.info("Loading article...")
    article = arxiv_utils.get_article_by_id(arxiv_id)
    if article is None:
        logger.error("Could not find article by this id!")
        return None
    article_df = arxiv_utils.build_arxiv_df([arxiv_utils.build_arxiv_dict(article)])
    # Loading database
    logger.debug("Loading Database...")
    all_papers_df = preprocessing.load_dataset()
    article_df = preprocessing.pipeline_arxiv_articles.transform(article_df)
    if encode_authors_topic:
        all_papers_df = feature_engineering.CreateAuthorsTopicsFeature().fit_transform(all_papers_df)
        article_df = feature_engineering.CreateAuthorsTopicsFeature().transform(article_df)
        cv = preprocessing.cvMixedEncoding
    else:
        cv = preprocessing.cvMixed

    # Check whether the article is already in our catalogue
    idx_dup = all_papers_df.loc[all_papers_df["id"] == article_df["id"].iloc[0]].index
    if not idx_dup.empty:
        all_papers_df = all_papers_df.drop(idx_dup)

    # Calculating similarities
    logger.info("Computing similarities...")
    concat_df = pd.concat([article_df, all_papers_df], ignore_index=True)
    sim_line, _ = recommender.build_similarity_line(concat_df, 0, cv)
    # Extracting recommended articles
    recommendations = recommender.get_similar_entries_line(concat_df, 0, sim_line, n_recommendations=n_recommendations)
    for _, row in recommendations.iterrows():
        print(
            f"https://arxiv.org/abs/{row['sim_id']} | {row['sim_entries_title']}, {preprocessing.undo_clean_author_list(row['sim_entries_authors'])}"
        )
    return recommendations
