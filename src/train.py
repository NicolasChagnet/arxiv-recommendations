from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from src import config, preprocessing, recommender
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
import logging
import pickle

logger = logging.getLogger("arxiv_recommender")

# Log to file
fh = logging.FileHandler(config.path_log)
fh.setLevel(logging.INFO)
logger.addHandler(fh)


def build_train_data(data_clean, cos_sim_mat):
    # Create target feature from library status and cosine similarity
    data_clean = preprocessing.BuildAverageSimilarity(cos_sim_mat).fit_transform(data_clean)
    return data_clean


def train_RFC():
    logger.info("Starting training...")

    # Load the data
    data_clean = preprocessing.load_dataset()
    # Build cosine similarity matrix
    logger.info("Computing similarity matrix...")
    cvTransf = preprocessing.get_cv(ngram_range=(2, 2), max_features=10_000)
    cos_sim_mat, X = recommender.build_similarity_matrix(
        data_clean[config.word_soup],
        cvTransf,
    )
    # Build training data
    data_target = preprocessing.BuildAverageSimilarity(cos_sim_mat).fit_transform(data_clean)
    y = data_target["target"]

    # Resample the data since it'll tend to be biased
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Cross validation
    rfc = RandomForestClassifier()
    scores_f1 = cross_validate(rfc, X_resampled, y_resampled, cv=5, scoring="f1_macro")
    logger.info("F1 scores on validation tests: %s", scores_f1["test_score"])

    # Fit to the entire training set
    rfc.fit(X_resampled, y_resampled)

    # Saving model to file
    logger.info("Saving model to file...")
    with open(config.path_model, "wb") as file:
        pickle.dump(rfc, file)

    # Saving transformer to file
    logger.info("Saving transformer to file...")
    with open(config.path_cv, "wb") as file:
        pickle.dump(cvTransf, file)
