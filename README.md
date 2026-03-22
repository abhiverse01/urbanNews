# Viral News Bot 📰

AI-powered news bot that fetches real news, scores virality, generates engaging posts,
and publishes them automatically 3× per day. Zero cost, zero maintenance.

---

## Architecture

```
Free News Sources (RSS + Guardian + NYT)
         ↓
  News Fetcher Agent
         ↓
  Viral Scorer Agent
         ↓
  DistilGPT-2 Generator  ← fine-tuned on Colab T4 (free), hosted on HF Hub
         ↓
  GitHub Actions Scheduler (cron, 3×/day, free)
         ↓
  Telegram Bot │ HF Streamlit UI │ Discord Webhook
```

---

## Setup Guide (< 1 hour, all free)

### Step 1 — Fine-tune the model on Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Copy-paste `colab_finetune.py` content into cells
4. Run all cells top to bottom (~25 minutes)
5. Your model is now at `https://huggingface.co/YOUR_USERNAME/viral-news-distilgpt2`

### Step 2 — Get free API keys

| Service | URL | Notes |
|---|---|---|
| Guardian API | https://open-platform.theguardian.com/access | Instant, free forever |
| NYT API | https://developer.nytimes.com | Free, 500 req/day |
| HuggingFace token | https://huggingface.co/settings/tokens | Create with **write** access |

### Step 3 — Create Telegram bot (for posting)

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` → follow prompts → copy the **bot token**
3. Create a public channel → add your bot as **Admin with post permissions**
4. Your channel ID: `@your_channel_name`

### Step 4 — Create Discord webhook (optional)

1. Discord Server → Settings → Integrations → Webhooks → New Webhook
2. Copy the webhook URL

### Step 5 — Set up GitHub repo

```bash
git init viral-news-bot
cd viral-news-bot
# Copy all files here
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/viral-news-bot
git push -u origin main
```

Add secrets at **Settings → Secrets and variables → Actions**:

| Secret name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From BotFather |
| `TELEGRAM_CHANNEL_ID` | `@your_channel` |
| `DISCORD_WEBHOOK_URL` | Webhook URL (optional) |
| `HF_TOKEN` | HuggingFace write token |
| `HF_DATASET_REPO` | `your-username/news-posts` |
| `HF_USERNAME` | Your HuggingFace username |
| `GUARDIAN_API_KEY` | Guardian key |
| `NYT_API_KEY` | NYT key (optional) |

### Step 6 — Create HuggingFace Dataset (data store)

1. Go to https://huggingface.co/new-dataset
2. Name it `news-posts`, set to **Public**
3. That's it — the pipeline writes to it automatically

### Step 7 — Deploy Streamlit UI on HuggingFace Spaces

1. https://huggingface.co/new-space
2. Space name: `viral-news-bot`, SDK: **Streamlit**
3. Paste `streamlit_app.py` content into `app.py`
4. Add `requirements.txt`
5. Set Space secret: `HF_DATASET_REPO = your-username/news-posts`
6. Your UI is live at `https://huggingface.co/spaces/your-username/viral-news-bot`

### Step 8 — Test the pipeline manually

GitHub repo → Actions tab → Viral News Bot → **Run workflow**

---

## Customisation

**Change posting times** — edit cron in `.github/workflows/news_bot.yml`:
```yaml
- cron: '15 2 * * *'   # adjust for your timezone
```

**Add more RSS feeds** — edit `RSS_FEEDS` in `app/fetch_news.py`

**Change number of daily posts** — edit `python main.py --n 2` in the workflow (max 3 recommended)

**Tune virality scoring** — edit weights in `app/score_viral.py`

---

## Cost breakdown

| Component | Cost |
|---|---|
| Google Colab (training) | Free (T4 GPU, one-time ~25 min) |
| HuggingFace Hub (model hosting) | Free |
| HuggingFace Spaces (UI) | Free |
| HuggingFace Datasets (storage) | Free |
| GitHub Actions (scheduler) | Free (uses ~450/2000 min/month) |
| Guardian API | Free |
| NYT API | Free |
| Telegram Bot | Free |
| Discord Webhook | Free |
| **Total** | **$0/month** |
