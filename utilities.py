import random as rnd
import urllib.request as libreq

import feedparser as fd


# Given a category, month and year, returns a random arXiv article
def get_arxiv_articles(cat, month, year):
    month_str = "{:02d}".format(month)
    query = f"http://export.arxiv.org/api/query?search_query=cat:{cat}+AND+submittedDate:[{year}{month_str}01+TO+{year}{month_str}31]"
    data = libreq.urlopen(query).read()
    feed = fd.parse(data)["entries"]
    try:
        return rnd.sample(feed, 10)
    except ValueError:
        return None


# List of categories to consider
cats = ["hep-th", "cond-mat.str-el", "gr-qc"]
years = range(2000, 2025)
months = range(1, 13)


# Get random article from random month, year and category
def get_random_articles():
    rnd_cat = rnd.choice(cats)
    rnd_year = rnd.choice(years)
    rnd_month = rnd.choice(months)

    ret_rnd = get_arxiv_articles(rnd_cat, rnd_month, rnd_year)
    if ret_rnd is None:
        return None
    return (rnd_year, get_arxiv_articles(rnd_cat, rnd_month, rnd_year))
