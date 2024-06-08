# Zotero and arXiv Recomendation System

#### -- Project Status: [ Active ]

## Project Intro/Objective
This project contains some code meant to analyse my Zotero library containg journal articles I have been amassing for many years. The objective is to build some scraping and recommendation system on the arXiv in order to find and classify new papers and help find relevant research tuned to my interests. This project can be viewed as an improvement over the basic scraper and keyword highlighter I previously developped: [arxiv_scanner_flask](https://github.com/NicolasChagnet/arxiv_scanner_flask)

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
- **Classifier**: I built a classifier model (using a basic Random Forest Classifier) in order to predict whether an article *should* be included in my library (hence whether it should be recommended) without having to recalculate the entire similarity matrix. To train the model, since the in-library subset is small and the datsaset is imbalanced, I used oversampling methods in order to rebalance the dataset.
- **Packaging** (TODO): This project can also pull new daily arXiv articles and use the trained model to extract which articles should be read in priority.
- **Re-training** (TODO): The underlying model can keep learning as new data is added daily and the use confirms addition to the library.


## Getting Started

TODO


<!-- ## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](#)
* [Notebook/Markdown/Slide DeckTitle](#)
* [Blog Post](#) -->
