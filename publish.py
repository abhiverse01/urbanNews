"""
publish.py
Publisher Agent.

Channels:
  • Telegram Bot (free — create one at t.me/BotFather)
  • Discord Webhook (free — Server Settings → Integrations → Webhooks)
  • HuggingFace Dataset (free — used as the Streamlit UI's data source)
"""
import os, json, logging
from datetime import datetime, timezone
import requests

log = logging.getLogger("publisher")

TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.environ.get("TELEGRAM_CHANNEL_ID", "")   # e.g. @your_channel or numeric id
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "your-hf-username/news-posts")

# ── Telegram ─────────────────────────────────────────────────
def post_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        log.info("Telegram not configured, skipping.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id":                  TELEGRAM_CHAT,
                "text":                     message[:4096],   # Telegram limit
                "parse_mode":               "HTML",
                "disable_web_page_preview": False,
            },
            timeout=10,
        )
        ok = r.status_code == 200
        log.info(f"Telegram: {'✅' if ok else '❌'} ({r.status_code})")
        return ok
    except Exception as e:
        log.warning(f"Telegram error: {e}")
        return False

# ── Discord ──────────────────────────────────────────────────
def post_discord(message: str) -> bool:
    if not DISCORD_WEBHOOK:
        log.info("Discord not configured, skipping.")
        return False
    try:
        r = requests.post(
            DISCORD_WEBHOOK,
            json={"content": message[:2000]},   # Discord limit
            timeout=10,
        )
        ok = r.status_code in (200, 204)
        log.info(f"Discord: {'✅' if ok else '❌'} ({r.status_code})")
        return ok
    except Exception as e:
        log.warning(f"Discord error: {e}")
        return False

# ── HuggingFace Dataset (Streamlit data source) ──────────────
def save_to_hf(posts: list[dict]) -> bool:
    """
    Append posts to a HuggingFace Dataset.
    The Streamlit app reads from this dataset.
    Keeps last 200 records to stay within free tier limits.
    """
    if not HF_TOKEN:
        log.info("HF_TOKEN not set, skipping dataset save.")
        _save_local_fallback(posts)
        return False

    try:
        import pandas as pd
        from datasets import load_dataset, Dataset

        timestamp = datetime.now(timezone.utc).isoformat()
        new_rows = [{
            "timestamp": timestamp,
            "title":     p["title"],
            "post":      p["post"],
            "source":    p["source"],
            "link":      p["link"],
        } for p in posts]

        # Load existing records
        try:
            existing = load_dataset(HF_DATASET_REPO, split="train")
            df_old   = existing.to_pandas()
        except Exception:
            df_old = pd.DataFrame()

        df = pd.concat([pd.DataFrame(new_rows), df_old], ignore_index=True)
        df = df.drop_duplicates(subset=["title"]).head(200)

        Dataset.from_pandas(df).push_to_hub(
            HF_DATASET_REPO, token=HF_TOKEN, private=False
        )
        log.info(f"HF Dataset updated: {len(new_rows)} new, {len(df)} total")
        return True

    except Exception as e:
        log.warning(f"HF Dataset error: {e}")
        _save_local_fallback(posts)
        return False

def _save_local_fallback(posts: list[dict]):
    """Save posts to a local JSON file when HF is unavailable."""
    path = "posts_history.json"
    try:
        try:
            with open(path) as f:
                history = json.load(f)
        except Exception:
            history = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for p in posts:
            history.insert(0, {"timestamp": timestamp, **p})
        with open(path, "w") as f:
            json.dump(history[:200], f, indent=2)
        log.info(f"Saved to local {path}")
    except Exception as e:
        log.warning(f"Local save error: {e}")

# ── Public API ────────────────────────────────────────────────
def publish_all(posts: list[dict]) -> dict:
    results = {"telegram": 0, "discord": 0, "hf": False}
    for p in posts:
        if post_telegram(p["post"]):
            results["telegram"] += 1
        if post_discord(p["post"]):
            results["discord"] += 1
    results["hf"] = save_to_hf(posts)
    log.info(f"Published: {results}")
    return results
