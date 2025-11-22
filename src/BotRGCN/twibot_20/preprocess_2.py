import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os

# ---------------- Load dataset ----------------
print("Loading dataset...")
user, tweet = fast_merge(dataset="Twibot-20")

# ---------------- Description extraction ----------------
user_text = []
for profile in user.get("profile", []):
    if isinstance(profile, dict):
        user_text.append(profile.get("description", None))
    else:
        user_text.append(None)

if len(user_text) == 0:
    user_text = [None]

# ---------------- Tweet extraction ----------------
tweet_text = []

# "tweet" column contains LISTS of strings
if "tweet" in tweet.columns:
    for t in tweet["tweet"]:
        if isinstance(t, list):
            tweet_text.extend(t)        # flatten tweet list
        else:
            tweet_text.append(None)
else:
    tweet_text = []

print(f"Total extracted tweet texts: {len(tweet_text)}")

# ---------------- Load each_user_tweets (PT FILE!!) ----------------
print("Loading per-user tweet indices...")
each_user_tweets_path = './processed_data/each_user_tweets.pt'

if os.path.exists(each_user_tweets_path):
    each_user_tweets = torch.load(each_user_tweets_path)
else:
    print("Error: each_user_tweets.pt not found. Run preprocess_1.py first.")
    exit(1)

# Guarantee correct structure
safe_user_tweets = []
for i in range(len(user)):
    safe_user_tweets.append(list(each_user_tweets.get(i, [])))
each_user_tweets = safe_user_tweets


# ---------------- Feature extraction pipeline ----------------
print("Initializing embedding model...")
device = 0 if torch.cuda.is_available() else -1
feature_extract = pipeline(
    "feature-extraction",
    model="roberta-base",
    tokenizer="roberta-base",
    device=device,
    padding=True,
    truncation=True,
    max_length=50
)

def fe(text):
    """Safe feature extraction"""
    try:
        out = feature_extract(text)
        if out and len(out[0]) > 0:
            return torch.tensor(out[0]).mean(dim=0)
    except:
        pass
    return torch.zeros(768)


# ---------------- User description embeddings ----------------
def Des_embbeding():
    print("Running user description embeddings...")
    path = "./processed_data/des_tensor.pt"

    if os.path.exists(path):
        print("Loaded existing user description embeddings.")
        return torch.load(path)

    des_vec = []
    for text in tqdm(user_text):
        if text is None or str(text).strip() == "":
            des_vec.append(torch.zeros(768))
        else:
            des_vec.append(fe(text))

    if len(des_vec) == 0:
        des_vec = [torch.zeros(768)]

    des_tensor = torch.stack(des_vec, dim=0)
    torch.save(des_tensor, path)
    print("Finished user description embeddings.")
    return des_tensor


# ---------------- Tweets embeddings ----------------
def tweets_embedding():
    print("Running tweets embeddings...")
    path = "./processed_data/tweets_tensor.pt"

    if os.path.exists(path):
        print("Loaded existing tweets embeddings.")
        return torch.load(path)

    tweet_vecs = []

    for user_tweet_indices in tqdm(each_user_tweets):
        if len(user_tweet_indices) == 0:
            tweet_vecs.append(torch.zeros(768))
            continue

        total = torch.zeros(768)
        count = 0

        for idx in user_tweet_indices:
            if idx < len(tweet_text):
                text = tweet_text[idx]
            else:
                text = None

            if text is None or str(text).strip() == "":
                vec = torch.zeros(768)
            else:
                vec = fe(text)

            total += vec
            count += 1

        if count == 0:
            tweet_vecs.append(torch.zeros(768))
        else:
            tweet_vecs.append(total / count)

    if len(tweet_vecs) == 0:
        tweet_vecs = [torch.zeros(768)]

    tweet_tensor = torch.stack(tweet_vecs, dim=0)
    torch.save(tweet_tensor, path)
    print("Finished tweets embeddings.")
    return tweet_tensor


# ---------------- Run everything ----------------
des_tensor = Des_embbeding()
tweet_tensor = tweets_embedding()
print("All embeddings generated successfully!")
