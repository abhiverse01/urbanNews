"""
score_viral.py
Viral Scorer Agent — ranks articles by engagement potential.
Pure heuristics: fast, zero-cost, surprisingly effective.
"""
import re, logging
from datetime import datetime, timezone

log = logging.getLogger("scorer")

# Weights: keyword appears in TITLE (higher) vs body (lower)
VIRAL_TITLE_KEYWORDS = {
    # urgency / exclusivity
    "breaking": 5, "exclusive": 4, "urgent": 4, "just in": 4,
    "leaked": 5, "revealed": 4, "exposed": 4, "secret": 3,
    # scale
    "massive": 3, "historic": 4, "unprecedented": 4, "record": 3,
    "first ever": 4, "biggest": 3, "largest": 3, "worst": 3,
    # emotional
    "shocking": 3, "alarming": 3, "crisis": 3, "collapse": 3,
    "surge": 2, "plunge": 2, "ban": 2, "warning": 2,
    # engagement bait (mild boost)
    "why": 1, "how": 1, "what": 1, "you need": 2, "must": 2,
    "everything": 1, "here's": 1,
}

HIGH_ENGAGEMENT_TOPICS = {
    "ai": 4, "artificial intelligence": 4, "chatgpt": 4, "openai": 4,
    "climate": 3, "ukraine": 3, "israel": 3, "gaza": 3, "iran": 3,
    "election": 4, "trump": 3, "elon musk": 3, "tesla": 2,
    "crypto": 3, "bitcoin": 3, "stock market": 3, "recession": 3,
    "earthquake": 4, "flood": 3, "hurricane": 3, "pandemic": 4,
    "cancer": 3, "cure": 3, "vaccine": 3, "virus": 3,
    "nasa": 3, "space": 2, "moon": 2, "mars": 2,
    "war": 4, "nuclear": 4, "missile": 3, "attack": 3,
}

SOURCE_AUTHORITY = {
    "Reuters":       4,
    "AP":            4,
    "BBC World":     3, "BBC Technology": 3,
    "The Guardian":  3,
    "New York Times":3,
    "NPR News":      2,
    "Al Jazeera":    2,
    "TechCrunch":    2,
    "NASA":          3,
    "Science Daily": 2,
}

def _recency_score(published: str) -> float:
    """Newer articles score higher. Max 10 for < 2 hours old."""
    if not published:
        return 0
    try:
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(published)
        except Exception:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        now   = datetime.now(timezone.utc)
        hours = max(0, (now - dt.astimezone(timezone.utc)).total_seconds() / 3600)
        if   hours < 1:  return 10.0
        elif hours < 2:  return  8.0
        elif hours < 6:  return  5.0
        elif hours < 12: return  3.0
        elif hours < 24: return  1.5
        else:            return  0.0
    except Exception:
        return 0.0

def compute_viral_score(article: dict) -> float:
    title  = article.get("title", "").lower()
    body   = (article.get("summary", "") or "").lower()
    full   = title + " " + body
    score  = 0.0

    # Viral keywords in title
    for kw, w in VIRAL_TITLE_KEYWORDS.items():
        if kw in title:
            score += w
        elif kw in body:
            score += w * 0.3

    # High-engagement topics
    for topic, w in HIGH_ENGAGEMENT_TOPICS.items():
        if topic in full:
            score += w

    # Question headline = engagement
    if "?" in article.get("title", ""):
        score += 2

    # Number in headline ("5 things", "100,000 people")
    if re.search(r'\b\d+\b', article.get("title", "")):
        score += 1.5

    # ALL CAPS word (BREAKING, URGENT…)
    if re.search(r'\b[A-Z]{4,}\b', article.get("title", "")):
        score += 2

    # Source authority
    score += SOURCE_AUTHORITY.get(article.get("source", ""), 0)

    # Recency
    score += _recency_score(article.get("published", ""))

    return round(score, 2)

def rank_articles(articles: list[dict], top_n: int = 3) -> list[dict]:
    scored = sorted(articles, key=lambda a: compute_viral_score(a), reverse=True)
    top = scored[:top_n]
    for a in top:
        log.info(f"[{compute_viral_score(a):.1f}] {a['title'][:80]}")
    return top
