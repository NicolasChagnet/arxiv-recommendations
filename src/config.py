from pathlib import Path

from pyprojroot.here import here  # This package is useful to track the root directory of the package

# Useful locations of files

path_root_folder = here()

path_log = path_root_folder / "out.log"
path_figs = path_root_folder / "figs"
path_data_dir = path_root_folder / "data"
path_data_raw = path_data_dir / "raw"
path_data_interim = path_data_dir / "interim"
path_data_wf = path_data_dir / "workflow"
path_data_merged = path_data_interim / "zotero_merged_data.csv"
path_data_notedited = path_data_interim / "zotero_data.csv"
path_data_edited = path_data_interim / "zotero_data_edited.csv"
path_data_author_encoding = path_data_wf / "encoding_authors.csv"

path_models = path_root_folder / "models"
path_model = path_models / "rfc.pkl"
path_cv = path_models / "cv.pkl"
path_sim_mat = path_models / "sim_mat.pkl"

# Useful quantities for the arXiv API scraping

arxiv_cats = ["hep-th", "cond-mat.stat-mech", "cond-mat.str-el", "gr-qc", "quant-ph"]
arxiv_years = range(1990, 2025)
arxiv_months = range(1, 13)

# Other parameters
threshold_similarity_normalized = 0.2
word_soup = "word_soup_title_categories"
