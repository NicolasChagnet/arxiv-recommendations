from src import config, train, arxiv_utils, application
import logging
import argparse

logger = logging.getLogger("arxiv_recommender")
logger.setLevel(logging.DEBUG)

# Log to file
fh = logging.FileHandler(config.path_log)
fh.setLevel(logging.INFO)
logger.addHandler(fh)

parser = argparse.ArgumentParser(description="Recommendation tools for the arXiv.")
parser.add_argument("id", type=str, help="The id of an arXiv paper.")
parser.add_argument("--encode_topics", help="Whether to use topic encoding for authors", action="store_true")
parser.add_argument("-n", "--number_recommendations", help="Number of recommendations to return", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    n_recommendations = args.number_recommendations if args.number_recommendations is not None else 15
    application.find_recommendations(
        args.id, n_recommendations=n_recommendations, encode_authors_topic=args.encode_topics
    )
