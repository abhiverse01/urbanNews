"""
streamlit_app.py
HuggingFace Spaces UI — displays generated viral posts in real time.
Deploy: New Space → Streamlit → paste this file → done.
The Space only DISPLAYS posts. The pipeline runs in GitHub Actions.
"""
import os
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "your-hf-username/news-posts")

st.set_page_config(
    page_title="Viral News Bot",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
.post-card {
    background: #f8f9fa;
    border-left: 4px solid #e63946;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.dark-mode .post-card { background: #1e1e2e; }
.post-meta { font-size: 0.78rem; color: #6c757d; margin-bottom: 0.4rem; }
.source-badge {
    display: inline-block;
    background: #e63946;
    color: white;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.72rem;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.title("📰 Viral News Bot")
    st.caption("AI-generated viral posts · Automated 3× daily via GitHub Actions")

with col_meta:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ── Load data ─────────────────────────────────────────────────
@st.cache_data(ttl=180)  # Refresh every 3 minutes
def load_posts():
    try:
        from datasets import load_dataset
        ds  = load_dataset(HF_DATASET_REPO, split="train")
        df  = ds.to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df  = df.dropna(subset=["timestamp"])
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

df = load_posts()

if "error" in df.columns:
    st.error(f"Could not load posts: {df['error'][0]}")
    st.info("The GitHub Actions pipeline may not have run yet. Check back in a few hours.")
    st.stop()

if df.empty:
    st.info("No posts yet. The bot generates posts 3× daily. Check back soon!")
    st.stop()

# ── Stats ─────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total posts",     len(df))
m2.metric("Sources covered", df["source"].nunique())
m3.metric("Last updated",    df["timestamp"].max().strftime("%b %d %H:%M UTC"))
now_utc = datetime.now(timezone.utc)
hours_ago = (now_utc - df["timestamp"].max().to_pydatetime()).total_seconds() / 3600
m4.metric("Hours since last post", f"{hours_ago:.1f}h")

st.divider()

# ── Filters ───────────────────────────────────────────────────
col_filter, col_search = st.columns([2, 3])
with col_filter:
    sources = ["All sources"] + sorted(df["source"].unique().tolist())
    selected_source = st.selectbox("Filter by source", sources)
with col_search:
    search = st.text_input("Search posts", placeholder="e.g. AI, climate, election…")

filtered = df.copy()
if selected_source != "All sources":
    filtered = filtered[filtered["source"] == selected_source]
if search:
    mask = (
        filtered["title"].str.contains(search, case=False, na=False) |
        filtered["post"].str.contains(search, case=False, na=False)
    )
    filtered = filtered[mask]

st.caption(f"Showing {len(filtered)} posts")
st.divider()

# ── Posts ─────────────────────────────────────────────────────
for _, row in filtered.iterrows():
    ts = row["timestamp"].strftime("%Y-%m-%d %H:%M UTC") if pd.notna(row["timestamp"]) else ""
    st.markdown(f"""
    <div class="post-card">
      <div class="post-meta">
        <span class="source-badge">{row.get('source','')}</span>
        {ts}
      </div>
      <strong>{row.get('title','')}</strong>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 View generated post", expanded=False):
        st.text_area(
            label="",
            value=row.get("post", ""),
            height=200,
            key=f"post_{_}",
            label_visibility="collapsed",
        )
        if row.get("link"):
            st.markdown(f"[🔗 Read original article]({row['link']})")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with DistilGPT-2 (fine-tuned) · "
    "Scheduled via GitHub Actions · "
    "Data stored on HuggingFace Datasets · "
    "100% free infrastructure"
)
