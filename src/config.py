from pathlib import Path

from pyprojroot.here import here  # This package is useful to track the root directory of the package

# Useful locations of files

path_root_folder = here()

path_data_dir = path_root_folder / "data"
path_data_raw = path_data_dir / "raw"
path_data_interim = path_data_dir / "interim"
path_data_merged = path_data_interim / "zotero_merged_data.csv"
path_data_notedited = path_data_interim / "zotero_data.csv"
path_data_edited = path_data_interim / "zotero_data_edited.csv"


# Useful quantities for the arXiv API scraping

arxiv_cats = ["hep-th", "cond-mat.stat-mech", "cond-mat.str-el", "gr-qc", "quant-ph"]
arxiv_years = range(1990, 2025)
arxiv_months = range(1, 13)
