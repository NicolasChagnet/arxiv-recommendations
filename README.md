# Zotero and arXiv Recomendation System

#### -- Project Status: [ Active ]

## Project Intro/Objective
This project contains some code meant to analyse my Zotero library containg journal articles I have been amassing for many years. The objective is to build some scraping and recommendation system on the arXiv in order to find and classify new papers and help find relevant research tuned to my interests. This project can be viewed as an improvement over the basic scraper and keyword highlighter I previously developped: [arxiv_scanner_flask](https://github.com/NicolasChagnet/arxiv_scanner_flask).

<!--
### Collaborators
|Name     |  Github Page   |  Personal Website  |
|---------|-----------------|--------------------|
|Nicolas Chagnet | [NicolasChagnet](https://github.com/NicolasChagnet)| [nicolaschagnet.github.io](https://nicolaschagnet.github.io)  | -->

### Methods Used
* Data Analysis
* Machine Learning
* Data Visualization
* Predictive Modeling
* Content-based recommendation system
* Natural Language Processing
* Web scraping

### Technologies
* Python
* Pandas, Scikit-learn, numpy

## Project Description
This project involved various critical aspects of data science and is meant as a training project to bring to production some useful product. The various interesting steps are:
- **ETL**: Merge my Zotero library and a random arXiv sample of papers (different schema).
- **Analysis**: Analyze the dataset of arXiv articles (included in my library or not) in order to identify trends in topics, authors.
- **Encoding**: A novel technical aspect I used this project to train myself on is encoding text-based features.
- **Recommender system**: Using the sparsely encoded title, author list and category list of the articles, I built and compared various cosine similarity matrices which then served as recommendation matrices. This is a simple yet very effective system.
- **Classifier**: I built an unsupervised clustering system for topics using the summary column and non-negative matrix factorization. Each author is then attributed a list of most frequent topics which will be used as encoding for authors. Therefore, similarity between authors will now mean similarity of interests instead of textual similarity.
- **Packaging**: This project can be ran as a standalone script with any arXiv identifier. The program will first pull the article from arXiv and then run the similarity pipeline before returning recommendations.


## Getting Started

The notebook dealing with the data merging and arXiv random sampling can be found in [this notebook](notebooks/0_dataset_filtering.ipynb). The proof of concept for the recommender system and the classifier can be found in [this notebook](notebooks/1_exploratory_data_analysis.ipynb). The improvement with target encoding for authors can be found in [this notebook](notebooks/2_build_target_encoding_dataset.ipynb).

To see how to use this code, just run `python3 main.py --help` or any of the command below:
```bash
# Get recommendations for an article
python3 main.py 2303.17685

# Change the number of recommendations
python3 main.py 2303.17685 -n 30

# Use the target encoding instead of basic encoding for authors (slower)
python3 main.py 2303.17685 --encode_topics
```


<!-- ## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](#)
* [Notebook/Markdown/Slide DeckTitle](#)
* [Blog Post](#) -->
