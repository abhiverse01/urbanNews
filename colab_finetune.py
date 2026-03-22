# ============================================================
# VIRAL NEWS BOT — DistilGPT-2 Fine-tuning (Google Colab)
# Run each section as a Colab cell.
# GPU: Runtime → Change runtime type → T4 GPU (free)
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
# !pip install -q transformers datasets accelerate feedparser huggingface_hub

# ── CELL 2: Collect training data from free RSS feeds ───────
import feedparser, json, random, re

FEEDS = {
    "BBC":       "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters":   "https://feeds.reuters.com/reuters/topNews",
    "Guardian":  "https://www.theguardian.com/world/rss",
    "NPR":       "https://feeds.npr.org/1001/rss.xml",
    "TechCrunch":"https://techcrunch.com/feed/",
    "AP":        "https://rsshub.app/apnews/topics/apf-topnews",
}

raw_articles = []
for name, url in FEEDS.items():
    try:
        feed = feedparser.parse(url)
        for e in feed.entries[:20]:
            t = re.sub(r'<[^>]+>', '', e.get("title", "")).strip()
            s = re.sub(r'<[^>]+>', '', e.get("summary", "")).strip()
            if t and len(t) > 20:
                raw_articles.append({"title": t, "summary": s[:200], "source": name})
    except Exception as ex:
        print(f"Skipped {name}: {ex}")

print(f"Fetched {len(raw_articles)} real headlines")

# ── CELL 3: Build training examples ─────────────────────────
VIRAL_FORMATS = [
    "TITLE: {title}\nPOST: 🔥 BREAKING: {title}\n\n{summary}\n\n#BreakingNews #Trending #MustRead\n<|endoftext|>",
    "TITLE: {title}\nPOST: ⚡ Just in from {source}:\n\n{title}\n\n{summary}\n\nWhat do you think? Drop a comment 👇\n\n#News #Viral\n<|endoftext|>",
    "TITLE: {title}\nPOST: 🚨 You need to see this:\n\n{title}\n\n{summary}\n\nShare before it disappears. 🔁\n\n#Breaking #MustRead\n<|endoftext|>",
    "TITLE: {title}\nPOST: 📰 Big story:\n\n{title}\n\n{summary}\n\n💡 Stay informed. Follow for updates.\n\n#TopNews #Trending\n<|endoftext|>",
    "TITLE: {title}\nPOST: 👀 Everyone's talking about this:\n\n{title}\n\nHere's what you need to know:\n{summary}\n\n#News #BreakingNews\n<|endoftext|>",
]

training_texts = []
for art in raw_articles:
    fmt = random.choice(VIRAL_FORMATS)
    training_texts.append(fmt.format(
        title=art["title"],
        summary=art["summary"] if art["summary"] else "Read the full story for more details.",
        source=art["source"]
    ))

# Pad dataset — repeat with shuffled formats for variety
for art in random.sample(raw_articles, min(len(raw_articles), 40)):
    for fmt in random.sample(VIRAL_FORMATS, 2):
        training_texts.append(fmt.format(
            title=art["title"],
            summary=art["summary"] if art["summary"] else "Developing story.",
            source=art["source"]
        ))

random.shuffle(training_texts)
print(f"Total training examples: {len(training_texts)}")

with open("/content/train.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(training_texts))

# ── CELL 4: Fine-tune DistilGPT-2 ───────────────────────────
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)
from datasets import Dataset
import torch

MODEL_ID = "distilgpt2"   # 82M params — trains fast on T4, ~25 min
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU (switch to T4 GPU!)'}")

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )

ds = Dataset.from_dict({"text": training_texts})
ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
ds = ds.train_test_split(test_size=0.1, seed=42)

training_args = TrainingArguments(
    output_dir="/content/viral-gpt2-checkpoints",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    learning_rate=5e-4,
    fp16=torch.cuda.is_available(),   # FP16 on T4 GPU
    logging_steps=30,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
print("✅ Training complete!")

# ── CELL 5: Quick test before pushing ───────────────────────
model.eval()
test_prompt = "TITLE: Scientists discover major climate breakthrough\nPOST:"
inputs = tokenizer.encode(test_prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
    )
sample = tokenizer.decode(out[0], skip_special_tokens=True)
print("\n── Sample output ──")
print(sample.split("POST:")[-1].strip())

# ── CELL 6: Push to Hugging Face Hub ────────────────────────
from huggingface_hub import login

# Get token at: https://huggingface.co/settings/tokens (write access)
HF_TOKEN    = "hf_YOUR_TOKEN_HERE"
HF_USERNAME = "your-hf-username"          # Change this
REPO_NAME   = "viral-news-distilgpt2"

login(token=HF_TOKEN)
model.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}", token=HF_TOKEN)
tokenizer.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}", token=HF_TOKEN)
print(f"✅ Pushed to https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
