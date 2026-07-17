import re
import string
import time

import json
import os
import nltk
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:

    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_resources()

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/special characters, and collapse spaces.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list:
    """Tokenize text, remove stopwords, and lemmatize each token.

    Args:
        text: Cleaned text string.

    Returns:
        List of lemmatized tokens.
    """
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 1]
    tokens = [_LEMMATIZER.lemmatize(tok) for tok in tokens]
    return tokens


def preprocess_definition(text: str) -> str:
    """Full preprocessing pipeline: clean -> tokenize -> remove stopwords -> lemmatize.

    Args:
        text: Raw organization definition text.

    Returns:
        Preprocessed text ready for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


_WIKI_HEADERS = {
    "User-Agent": "SectorClassifier/1.0 (educational project; contact: none)"
}
_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "definitions.json")

COMPANY_SUFFIXES = {
    "limited",
    "ltd",
    "inc",
    "corp",
    "corporation",
    "company",
    "co",
    "private",
    "pvt",
    "plc",
    "llc"
}
KNOWN_NAMES = {
    "L&T": "Larsen and Toubro",
    "Larsen & Toubro": "Larsen and Toubro",
    "TCS": "Tata Consultancy Services",
    "SBI": "State Bank of India",
    "LIC": "Life Insurance Corporation",
    "ONGC": "Oil and Natural Gas Corporation",
    "IOC": "Indian Oil Corporation",
    "BPCL": "Bharat Petroleum",
    "HPCL": "Hindustan Petroleum",
}
def clean_org_name(name: str) -> str:
    """
    Clean organization names before searching Wikipedia.
    """

    if not isinstance(name, str):
        return ""

    name = name.replace("&", "and")

    name = name.replace(".", " ")

    name = re.sub(r"\s+", " ", name)

    words = name.split()

    cleaned = [
        w for w in words
        if w.lower() not in COMPANY_SUFFIXES
    ]

    return " ".join(cleaned).strip()
def load_cache():

    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(CACHE_FILE):
        return {}

    try:

        with open(CACHE_FILE, "r", encoding="utf8") as f:
            return json.load(f)

    except Exception:

        return {}
def save_cache(cache):

    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(CACHE_FILE, "w", encoding="utf8") as f:
        json.dump(cache, f, indent=4)

def _wiki_summary(title: str, timeout: int = 10) -> str:
    """Fetch the plain-text summary for an exact Wikipedia page title.

    Args:
        title: Wikipedia page title.
        timeout: Request timeout in seconds.

    Returns:
        The page's summary extract, or an empty string if unavailable.
    """
    url = _WIKI_SUMMARY_URL.format(title=requests.utils.quote(title))
    resp = requests.get(url, headers=_WIKI_HEADERS, timeout=timeout)
    if resp.status_code != 200:
        return ""
    data = resp.json()
    return data.get("extract", "") or ""


def _wiki_search_best_title(query: str, timeout: int = 10) -> str:
    """Search Wikipedia and return the title of the best-matching page.

    Args:
        query: Search query (typically an organization name).
        timeout: Request timeout in seconds.

    Returns:
        Best-matching page title, or an empty string if no match found.
    """
    params = {"q": query, "limit": "1"}
    resp = requests.get(
        _WIKI_SEARCH_URL, headers=_WIKI_HEADERS, params=params, timeout=timeout
    )
    if resp.status_code != 200:
        return ""
    data = resp.json()
    pages = data.get("pages", [])
    if not pages:
        return ""
    return pages[0].get("title", "")


def fetch_definition_from_web(org_name: str, delay: float = 0.5) -> dict:
    """
    Fetch an organization's definition from Wikipedia using
    multiple search strategies with caching.
    """

    if not isinstance(org_name, str) or not org_name.strip():
        return {"definition": "", "source": "not_found"}

    org_name = org_name.strip()
    org_name = KNOWN_NAMES.get(org_name, org_name)

    # --------------------------
    # Check local cache first
    # --------------------------

    cache = load_cache()

    if org_name in cache:
        return {
            "definition": cache[org_name],
            "source": "cache"
        }

    cleaned = clean_org_name(org_name)

    search_queries = [

    org_name,

    cleaned,
    cleaned + " company",
    cleaned + " india",
    cleaned + " organization",
    cleaned + " corporation",
    cleaned + " limited",
    cleaned + " plc",
]

    searched = set()

    for query in search_queries:

        if not query.strip():
            continue

        if query.lower() in searched:
            continue

        searched.add(query.lower())

        try:

            # Try exact page first

            summary = _wiki_summary(query)

            if summary:

                cache[org_name] = summary
                save_cache(cache)

                time.sleep(delay)

                return {

                    "definition": summary,

                    "source": "wikipedia_exact"

                }

            # Search Wikipedia

            best_title = _wiki_search_best_title(query)
            if best_title:

                best_title = best_title.replace("&", "and")
            if best_title:

                summary = _wiki_summary(best_title)

                if summary:

                    cache[org_name] = summary
                    save_cache(cache)

                    time.sleep(delay)

                    return {

                        "definition": summary,

                        "source": "wikipedia_search"

                    }

        except requests.RequestException:

            continue

        except Exception:

            continue

    return {

        "definition": "",

        "source": "not_found"

    }

    org_name = org_name.strip()

    try:
        summary = _wiki_summary(org_name)
        if summary:
            time.sleep(delay)
            return {"definition": summary, "source": "wikipedia_direct"}

        best_title = _wiki_search_best_title(org_name)
        if best_title:
            summary = _wiki_summary(best_title)
            time.sleep(delay)
            if summary:
                return {"definition": summary, "source": "wikipedia_search"}

        time.sleep(delay)
        return {"definition": "", "source": "not_found"}

    except requests.RequestException:
        return {"definition": "", "source": "error"}