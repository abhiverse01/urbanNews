"""
generate_post.py
Post Generator Agent.

Primary:  HuggingFace Inference API (free, no GPU needed in GitHub Actions)
Fallback: template-based generation (works even if model is cold/unavailable)
"""
import os, re, random, time, logging
import requests

log = logging.getLogger("generator")

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
MODEL_REPO  = f"{HF_USERNAME}/viral-news-distilgpt2"
INFERENCE_URL = f"https://api-inference.huggingface.co/models/{MODEL_REPO}"

# ── Templates (fallback + variety) ──────────────────────────
TEMPLATES = [
    "🔥 BREAKING: {title}\n\n{summary}\n\nThis is developing fast — stay tuned for updates.\n\n#BreakingNews #Trending",
    "⚡ Just in from {source}:\n\n{title}\n\n{summary}\n\nWhat are your thoughts? 👇\n\n#News #Viral",
    "🚨 You need to see this:\n\n{title}\n\n{summary}\n\nShare before it disappears. 🔁\n\n#Breaking #MustRead",
    "📰 Big story from {source}:\n\n{title}\n\n{summary}\n\n💡 Stay informed — follow for daily updates.\n\n#TopNews",
    "👀 Everyone's talking about this:\n\n{title}\n\nHere's what you need to know: {summary}\n\n#News #BreakingNews",
    "🌍 {source} reports:\n\n{title}\n\n{summary}\n\nLet us know what you think in the comments.\n\n#WorldNews #Trending",
]

def _template_post(article: dict) -> str:
    template = random.choice(TEMPLATES)
    summary = article.get("summary", "Read the full story for details.")
    if len(summary) > 200:
        # Break at last sentence boundary under 200 chars
        summary = re.sub(r'\s+\S+$', '...', summary[:200])
    return template.format(
        title   = article.get("title", ""),
        summary = summary,
        source  = article.get("source", "News"),
        link    = article.get("link", ""),
    )

def _hf_inference_post(article: dict, max_retries: int = 3) -> str | None:
    """Call HuggingFace Inference API with the fine-tuned model."""
    if not HF_TOKEN:
        return None

    prompt = (
        f"TITLE: {article['title']}\n"
        f"POST:"
    )
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 120,
            "do_sample": True,
            "temperature": 0.82,
            "top_p": 0.92,
            "repetition_penalty": 1.3,
            "return_full_text": False,
        },
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(INFERENCE_URL, headers=headers, json=payload, timeout=30)
            if r.status_code == 503:
                # Model is loading (cold start) — wait and retry
                log.info(f"Model loading, waiting 20s… (attempt {attempt+1})")
                time.sleep(20)
                continue
            if r.status_code != 200:
                log.warning(f"HF API returned {r.status_code}: {r.text[:100]}")
                return None
            generated = r.json()
            if isinstance(generated, list) and generated:
                text = generated[0].get("generated_text", "")
                # Clean: take up to end of first 2 paragraphs
                text = text.split("<|endoftext|>")[0].strip()
                lines = [l for l in text.split("\n") if l.strip()]
                return "\n\n".join(lines[:6])  # max 6 lines
        except Exception as e:
            log.warning(f"HF Inference error: {e}")
            time.sleep(5)

    return None

def create_viral_post(article: dict) -> str:
    # Try fine-tuned model first
    generated = _hf_inference_post(article)
    if generated and len(generated) > 40:
        # Append hashtags if model didn't add them
        if "#" not in generated:
            generated += "\n\n#BreakingNews #Trending #News"
        if article.get("link"):
            generated += f"\n\n🔗 {article['link']}"
        return generated

    # Fallback: deterministic template (always works)
    log.info("Using template fallback for generation")
    post = _template_post(article)
    if article.get("link"):
        post += f"\n\n🔗 {article['link']}"
    return post

def generate_posts(articles: list[dict]) -> list[dict]:
    posts = []
    for article in articles:
        post_text = create_viral_post(article)
        posts.append({
            "post":    post_text,
            "title":   article.get("title", ""),
            "source":  article.get("source", ""),
            "link":    article.get("link", ""),
        })
        log.info(f"Generated post for: {article['title'][:60]}")
    return posts
