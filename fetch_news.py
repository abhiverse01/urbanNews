"""
fetch_news.py
News Fetcher Agent — pulls from free RSS feeds + Guardian / NYT APIs.
No NewsAPI (production tier is paid). All sources here are genuinely free.
"""
import os, re, time, hashlib, logging
import feedparser
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fetcher")

# ── Free RSS feeds (no API key needed, no rate limits) ──────
RSS_FEEDS = {
    "BBC World":         "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Technology":    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "Reuters World":     "https://feeds.reuters.com/reuters/worldnews",
    "Reuters Business":  "https://feeds.reuters.com/reuters/businessnews",
    "The Guardian":      "https://www.theguardian.com/world/rss",
    "Al Jazeera":        "https://www.aljazeera.com/xml/rss/all.xml",
    "NPR News":          "https://feeds.npr.org/1001/rss.xml",
    "TechCrunch":        "https://techcrunch.com/feed/",
    "Science Daily":     "https://www.sciencedaily.com/rss/all.xml",
    "NASA":              "https://www.nasa.gov/rss/dyn/breaking_news.rss",
}

GUARDIAN_KEY = os.environ.get("GUARDIAN_API_KEY", "")   # free at open-platform.theguardian.com
NYT_KEY      = os.environ.get("NYT_API_KEY", "")         # free at developer.nytimes.com

def _clean(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r'<[^>]+>', ' ', text or "")
    return re.sub(r'\s+', ' ', text).strip()

def _article_id(title: str) -> str:
    return hashlib.md5(title.lower().encode()).hexdigest()[:12]

# ── RSS ──────────────────────────────────────────────────────
def fetch_rss(max_per_feed: int = 8) -> list[dict]:
    articles = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            for entry in feed.entries[:max_per_feed]:
                title   = _clean(entry.get("title", ""))
                summary = _clean(entry.get("summary", entry.get("description", "")))
                link    = entry.get("link", "")
                if len(title) < 20:
                    continue
                articles.append({
                    "id":      _article_id(title),
                    "title":   title,
                    "summary": summary[:300],
                    "link":    link,
                    "source":  source,
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            log.warning(f"RSS error [{source}]: {e}")
    log.info(f"RSS: {len(articles)} articles")
    return articles

# ── Guardian API (free, 12 calls/sec, no expiry) ─────────────
def fetch_guardian(query: str = "world news", n: int = 10) -> list[dict]:
    if not GUARDIAN_KEY:
        log.info("GUARDIAN_API_KEY not set, skipping.")
        return []
    try:
        r = requests.get(
            "https://content.guardianapis.com/search",
            params={
                "api-key":    GUARDIAN_KEY,
                "show-fields":"headline,trailText",
                "page-size":  n,
                "order-by":   "newest",
                "q":          query,
            },
            timeout=10,
        )
        results = r.json().get("response", {}).get("results", [])
        return [{
            "id":      _article_id(item.get("webTitle", "")),
            "title":   _clean(item.get("webTitle", "")),
            "summary": _clean(item.get("fields", {}).get("trailText", "")),
            "link":    item.get("webUrl", ""),
            "source":  "The Guardian",
            "published": item.get("webPublicationDate", ""),
        } for item in results if item.get("webTitle")]
    except Exception as e:
        log.warning(f"Guardian API error: {e}")
        return []

# ── NYT API (free, 500/day, 10/min) ─────────────────────────
def fetch_nyt(query: str = "world", n: int = 10) -> list[dict]:
    if not NYT_KEY:
        log.info("NYT_API_KEY not set, skipping.")
        return []
    try:
        r = requests.get(
            "https://api.nytimes.com/svc/search/v2/articlesearch.json",
            params={
                "api-key": NYT_KEY,
                "q":       query,
                "sort":    "newest",
                "fl":      "headline,abstract,web_url,pub_date,source",
            },
            timeout=10,
        )
        docs = r.json().get("response", {}).get("docs", [])[:n]
        return [{
            "id":      _article_id(d.get("headline", {}).get("main", "")),
            "title":   _clean(d.get("headline", {}).get("main", "")),
            "summary": _clean(d.get("abstract", "")),
            "link":    d.get("web_url", ""),
            "source":  "New York Times",
            "published": d.get("pub_date", ""),
        } for d in docs if d.get("headline", {}).get("main")]
    except Exception as e:
        log.warning(f"NYT API error: {e}")
        return []

# ── Deduplicate ──────────────────────────────────────────────
def deduplicate(articles: list[dict]) -> list[dict]:
    seen_ids, seen_titles, unique = set(), set(), []
    for a in articles:
        title_key = a["title"][:40].lower()
        if a["id"] not in seen_ids and title_key not in seen_titles:
            seen_ids.add(a["id"])
            seen_titles.add(title_key)
            unique.append(a)
    return unique

# ── Public API ───────────────────────────────────────────────
def get_all_news() -> list[dict]:
    rss      = fetch_rss()
    guardian = fetch_guardian()
    nyt      = fetch_nyt()
    all_news = deduplicate(rss + guardian + nyt)
    log.info(f"Total unique articles: {len(all_news)}")
    return all_news
