import random as rnd
import re
import urllib.request as libreq
from urllib.parse import quote
from urllib.error import HTTPError
import numpy as np
import feedparser as fd
import pandas as pd
import requests
from bs4 import BeautifulSoup
from src import config
import time


def query_arxiv(query):
    """Queries the arXiv API

    Args:
        query (str): Query to submit.

    Returns:
        dict | None: Entries returned or None if an error is raised.
    """
    try:
        data = libreq.urlopen(query).read()
        feed = fd.parse(data)["entries"]
        time.sleep(3)  # Wait 3s after any API request per terms of use
        return feed
    except HTTPError:
        return None


def get_arxiv_id_by_title(title):
    """Given an arXiv article's title, finds the most relevant match by querying the API
    and extract its identifier from the link.

    Args:
        title (str): Article title.

    Returns:
        str | numpy.nan: arXiv identifier or NaN.
    """
    query = f'http://export.arxiv.org/api/query?search_query=ti:"{quote(title)}"&sortBy=relevance&max_results=1'
    entries = query_arxiv(query)
    if entries is None or len(entries) == 0:
        return np.nan
    arxiv_id = get_arxiv_id_from_link(entries[0]["link"])
    if arxiv_id is None:
        return np.nan
    return arxiv_id


def get_article_by_id(id):
    """Fetches a specific arXiv article by ID through the API.

    Args:
        id (str): arXiv id to fetch (format XXXX.XXXX or cat/XXXX)

    Returns:
        dict | None: Matching arXiv entry if found.
    """
    query = f"http://export.arxiv.org/api/query?id_list={id}"
    entries = query_arxiv(query)
    if entries is None or len(entries) == 0:
        return None
    return entries[0]


def get_article_by_title(title):
    """Fetches a specific arXiv article by title through the API.

    Args:
        title (str): Article title.

    Returns:
        dict | None: Matching arXiv entry if found.
    """
    query = f'http://export.arxiv.org/api/query?search_query=ti:"{quote(title)}"&sortBy=relevance&max_results=1'

    entries = query_arxiv(query)
    if entries is None or len(entries) == 0:
        return None
    return entries[0]


def get_random_articles_arxiv(cat, year, n_articles=10):
    """Given a category and year, queries the arXiv API and returns random articles.

    Args:
        cat (str): arXiv category of interest.
        year (int): Year selection.
        n_articles (int, optional): Number of articles to return from the query. Defaults to 10.

    Returns:
        list(dict) | None: List of random articles obtained through the query.
    """
    month = rnd.choice(range(1, 13))
    query = f"http://export.arxiv.org/api/query?max_results=2000\
&search_query=cat:{cat}+AND+submittedDate:[{year}{month:02d}01+TO+{year}{month:02d}31]"
    entries = query_arxiv(query)

    if entries is None:
        return None
    if len(entries) < n_articles:
        return entries
    return rnd.sample(entries, n_articles)


# def get_random_articles_arxiv_monthyear(month, year, n_articles=10):
#     """Given a month and year, queries the arXiv API and returns 10 random articles in a random category.

#     Args:
#         month (int): Month selection.
#         year (int): Year selection.
#         n_articles (int, optional): Number of articles to return from the query. Defaults to 10.

#     Returns:
#         list(dict) | None: List of random articles obtained through the query.
#     """
#     month_str = "{:02d}".format(month)
#     query = (
#         f"http://export.arxiv.org/api/query?search_query=submittedDate:[{year}{month_str}01+TO+{year}{month_str}31]"
#     )
#     entries = query_arxiv(query)
#     if entries is None:
#         return None
#     return rnd.sample(entries, n_articles)


def get_random_articles(n_articles=10):
    """Queries the arXiv API and returns random articles in randomly selected year and category among a pre-defined range.

    Parameters:
        n_articles (int, optional): Number of articles to return from the query. Defaults to 10.

    Returns:
        (int, list(dict)) | None: List of random articles obtained through the query with the selected year.
    """
    rnd_cat = rnd.choice(config.arxiv_cats)
    rnd_year = rnd.choice(config.arxiv_years)

    ret_rnd = get_random_articles_arxiv(rnd_cat, rnd_year, n_articles)
    if ret_rnd is None:
        return []
    return [build_arxiv_dict(entry) for entry in ret_rnd if entry is not None]


# def get_random_articles_random_categories(n_articles=10):
#     """Queries the arXiv API and returns random articles in randomly selected year among a pre-defined range and with a fully random category.

#     Returns:
#         (int, list(dict)) | None: List of random articles obtained through the query with the selected year.
#         n_articles (int, optional): Number of articles to return from the query. Defaults to 10.
#     """
#     rnd_year = rnd.choice(config.arxiv_years)
#     rnd_cat = rnd.choice(config.arxiv_cats)
#     ret_rnd = get_random_articles_arxiv(rnd_year, n_articles=n_articles)
#     if ret_rnd is None:
#         return None
#     return (rnd_year, ret_rnd)


def get_all_article_by_id_or_title(df):
    """Given a dataframe of articles, returns a standardized dataframe of arXiv articles either using the arxiv_id or the title.

    Args:
        df (pandas.DataFrame): DataFrame containing article information. Must have a column "Title" and a column "arxiv_id".

    Returns:
        pandas.DataFrame: DataFrame containing standardized arXiv information.
    """
    entries = [
        get_article_by_title(row["Title"]) if pd.isna(row["arxiv_id"]) else get_article_by_id(row["arxiv_id"])
        for idx, row in df.iterrows()
    ]
    entries_standardized = [build_arxiv_dict(entry) for entry in entries if entry is not None]
    return build_arxiv_df(entries_standardized)


def scrape_categories():
    """Scrapes all <h4> content to extract the list of all arXiv categories.

    Returns:
        list(str): List of all arXiv categories.
    """
    url_page = "https://arxiv.org/category_taxonomy"
    page = requests.get(url_page)

    soup = BeautifulSoup(page.content, "html.parser")
    cats_tags = soup.find_all("h4")
    cats = ["".join(element.findAll(text=True, recursive=False)) for element in cats_tags]
    return cats[1:]


def clean_string(s):
    """Cleans a string of newlines and multiple spaces

    Args:
        s (str): String to clean.

    Returns:
        str: Input string with no newline and no duplicate whitespaces.
    """
    s = s.replace("\n", " ")
    s = re.sub(r"\s\s+", " ", s)
    s = s.strip()
    return s


def build_arxiv_dict(feed_entry):
    """Converts the arXiv entry from API requests into a standardize dictionary.

    Args:
        feed_entry (dict): Entry from the API request.

    Returns:
        dict: Standardized output.
    """
    return {
        "id": get_arxiv_id_from_link(feed_entry["link"]),
        "title": clean_string(feed_entry["title"]),
        "authors": " ; ".join([author["name"] for author in feed_entry["authors"]]),
        "primary_category": feed_entry["arxiv_primary_category"]["term"],
        "categories": " ".join([tag["term"] for tag in feed_entry["tags"]]),
        "summary": clean_string(feed_entry["summary"]),
        "published": feed_entry["published"],
        "doi": feed_entry["arxiv_doi"] if "arxiv_doi" in feed_entry.keys() else "",
    }


def get_arxiv_id_from_link(link):
    """Given an arXiv link, extracts the identifier.

    Args:
        link (str): arXiv url

    Returns:
        str | None: arXiv identifier if a match is found
    """
    match_id = re.match(r"https?:\/\/arxiv.org\/abs\/([\d.]+|[a-z\-]+\/[\d]+)(v[0-9])?", link)
    if match_id:
        return match_id.group(1)
    return None


def build_arxiv_df(list_dicts):
    """Converts an API query from arXiv into a pandas DataFrame.

    Args:
        list_dicts (list(dict)): List of standardized arXiv entries.

    Returns:
        pandas.DataFrame: DataFrame containing all queried arXiv entries.
    """
    df = pd.DataFrame(list_dicts)
    if df.empty:
        return df
    # Parse dates
    df["published"] = pd.to_datetime(df["published"])

    df["doi"] = df.apply(lambda x: "10.48550/" + x["id"] if x["doi"] == "" else x["doi"], axis=1)

    return df


def get_arxiv_categories():
    """Converts the scraping result of all arXiv categories into a standardized version.
    Removes subcategories which are not of interest.

    Returns:
        list(str): List of standardized arXiv categories.
    """
    arXiv_categories_set = set(scrape_categories())
    set_prim = set()
    for cat in arXiv_categories_set:
        if "." in cat:
            set_prim.add(cat.split(".")[0])
    arXiv_categories_set.update(set_prim)
    return [x.strip() for x in list(arXiv_categories_set)]


def format_authors(authors):
    """Clean up the list of authors in an article category.

    Args:
        authors (list(str)): List of authors in an arXiv entry.

    Returns:
        list(str): Cleaned list of authors.
    """
    l_authors = authors.split("; ")
    return "; ".join([" ".join(author.split(",")[::-1]).strip() for author in l_authors])


def extract_arxiv_identifier(extra):
    """Extracts the arXiv identifier from Zotero's extra field

    Args:
        extra (str): Extra field
    """
    if type(extra) is not str or len(extra) == 0:
        return np.nan
    match_id = re.search(r"arXiv:\s?([^\s]+)\b", extra)
    if not match_id:
        return np.nan
    return match_id.group(1)
