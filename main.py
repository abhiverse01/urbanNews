"""
main.py
Pipeline Orchestrator — ties all agents together.
Called by GitHub Actions 3× per day.
"""
import logging, sys, json
from fetch_news    import get_all_news
from score_viral   import rank_articles
from generate_post import generate_posts
from publish       import publish_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

def run(n_posts: int = 2):
    log.info("═══ Viral News Bot pipeline starting ═══")

    # Step 1: Fetch
    log.info("STEP 1 — Fetching news from all sources…")
    articles = get_all_news()
    if not articles:
        log.error("No articles fetched. Check network / API keys.")
        sys.exit(1)

    # Step 2: Score & rank
    log.info(f"STEP 2 — Scoring {len(articles)} articles…")
    top_articles = rank_articles(articles, top_n=n_posts)

    # Step 3: Generate
    log.info(f"STEP 3 — Generating {n_posts} viral posts…")
    posts = generate_posts(top_articles)

    # Step 4: Publish
    log.info("STEP 4 — Publishing…")
    results = publish_all(posts)

    log.info("═══ Pipeline complete ═══")
    log.info(json.dumps(results, indent=2))
    return posts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2, help="Number of posts to generate")
    args = parser.parse_args()
    run(n_posts=args.n)
