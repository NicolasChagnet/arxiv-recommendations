import random as rnd
import re
import urllib.request as libreq

import feedparser as fd
import pandas as pd
import requests
from bs4 import BeautifulSoup


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


# Given month and year, returns a random arXiv article
def get_arxiv_articles_monthyear(month, year):
    month_str = "{:02d}".format(month)
    query = f"http://export.arxiv.org/api/query?search_query=submittedDate:[{year}{month_str}01+TO+{year}{month_str}31]"
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
    return (rnd_year, ret_rnd)


# Get random article from random month, year
def get_articles_random_categories():
    rnd_year = rnd.choice(years)
    rnd_month = rnd.choice(months)

    ret_rnd = get_arxiv_articles_monthyear(rnd_month, rnd_year)
    if ret_rnd is None:
        return None
    return (rnd_year, ret_rnd)


# Scrapes all <h4> content to extract the arXiv categories
def scrape_categories():
    url_page = "https://arxiv.org/category_taxonomy"
    page = requests.get(url_page)

    soup = BeautifulSoup(page.content, "html.parser")
    cats_tags = soup.find_all("h4")
    cats = [
        "".join(element.findAll(text=True, recursive=False)) for element in cats_tags
    ]
    return cats[1:]


# Fetches an article by id
def get_article_by_id(id):
    query = f"http://export.arxiv.org/api/query?id_list={id}"
    data = libreq.urlopen(query).read()
    feed = fd.parse(data)["entries"]
    try:
        return feed[0]
    except ValueError:
        return None


def clean_string(s):
    return re.sub("\s\s+", " ", s.replace("\n", ""))


def build_arxiv_dict(feed_entry):
    return {
        "title": clean_string(feed_entry["title"]),
        "authors": " ; ".join([author["name"] for author in feed_entry["authors"]]),
        "primary_category": feed_entry["arxiv_primary_category"]["term"],
        "categories": " ; ".join([tag["term"] for tag in feed_entry["tags"]]),
        "summary": clean_string(feed_entry["summary"]),
        "updated": feed_entry["updated"],
        "published": feed_entry["published"],
    }


def get_arxiv_categories():
    arXiv_categories_set = set(scrape_categories())
    set_prim = set()
    for cat in arXiv_categories_set:
        if "." in cat:
            set_prim.add(cat.split(".")[0])
    arXiv_categories_set.update(set_prim)
    return [x.strip() for x in list(arXiv_categories_set)]


def build_arxiv_df(list_dicts, categories):
    # Create dataframe
    df = pd.DataFrame(list_dicts)

    # Parse dates
    df["published"] = pd.to_datetime(df["published"])
    df["updated"] = pd.to_datetime(df["updated"])

    cols = [df]
    for i, cat in enumerate(categories):
        cols.append(df["categories"].apply(lambda x: cat in x))
        cols[i].name = "is_" + cat

    return pd.concat(cols, axis=1)
