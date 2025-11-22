import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os

# ---------------- Load dataset ----------------
print("Loading dataset...")
user, tweet = fast_merge(dataset="Twibot-20")

# Extract user descriptions safely
user_text = []
if 'profile' in user.columns:
    for profile in user['profile']:
        if isinstance(profile, dict):
            desc = profile.get('description', None)
            user_text.append(desc if desc is not None else None)
        else:
            user_text.append(None)
else:
    # fallback if profile column missing
    user_text = [None] * len(user)

# fallback: ensure user_text is not empty
if len(user_text) == 0:
    user_text = [None]

# Extract tweets safely
tweet_text = []
if 'tweet' in tweet.columns:
    for t in tweet['tweet']:
        tweet_text.append(t if isinstance(t, str) else None)
else:
    tweet_text = []

# Load each_user_tweets
print("Loading per-user tweet indices...")
each_user_tweets_path = './processed_data/each_user_tweets.npy'
if os.path.exists(each_user_tweets_path):
    each_user_tweets = np.load(each_user_tweets_path, allow_pickle=True).item()
    # Ensure a list for each user
    each_user_tweets = [list(each_user_tweets.get(i, [])) for i in range(len(user))]
else:
    print("Error: each_user_tweets.npy not found. Run preprocess_1.py first.")
    exit(1)

# ---------------- Feature extraction ----------------
print("Initializing feature extraction pipeline...")
device = 0 if torch.cuda.is_available() else -1
feature_extract = pipeline(
    'feature-extraction',
    model='roberta-base',
    tokenizer='roberta-base',
    device=device,
    padding=True,
    truncation=True,
    max_length=50,
    add_special_tokens=True
)

# ---------------- User description embeddings ----------------
def Des_embbeding():
    print('Running user description embeddings...')
    path = "./processed_data/des_tensor.pt"
    if os.path.exists(path):
        des_tensor = torch.load(path)
        print('Loaded existing user description embeddings.')
        return des_tensor

    des_vec = []
    for each in tqdm(user_text):
        if each is None or str(each).strip() == '':
            des_vec.append(torch.zeros(768))
        else:
            try:
                feature = torch.tensor(feature_extract(each))
                if feature is not None and len(feature) > 0 and len(feature[0]) > 0:
                    # mean over tokens
                    token_mean = feature[0].mean(dim=0)
                    des_vec.append(token_mean)
                else:
                    des_vec.append(torch.zeros(768))
            except Exception:
                des_vec.append(torch.zeros(768))

    # fallback safety
    if len(des_vec) == 0:
        des_vec = [torch.zeros(768)]

    des_tensor = torch.stack(des_vec, 0)
    torch.save(des_tensor, path)
    print('Finished user description embeddings.')
    return des_tensor

# ---------------- Tweets embeddings ----------------
def tweets_embedding():
    print('Running tweets embeddings...')
    path = "./processed_data/tweets_tensor.pt"
    if os.path.exists(path):
        tweet_tensor = torch.load(path)
        print('Loaded existing tweets embeddings.')
        return tweet_tensor

    tweets_list = []
    for user_tweets in tqdm(each_user_tweets):
        if len(user_tweets) == 0:
            tweets_list.append(torch.zeros(768))
            continue

        total_each_person_tweets = torch.zeros(768)
        count = 0
        for j, tweet_idx in enumerate(user_tweets):
            each_tweet = tweet_text[tweet_idx] if tweet_idx < len(tweet_text) else None
            if each_tweet is None or str(each_tweet).strip() == '':
                total_each_person_tweets += torch.zeros(768)
            else:
                try:
                    tweet_tensor = torch.tensor(feature_extract(each_tweet))
                    if tweet_tensor is not None and len(tweet_tensor) > 0 and len(tweet_tensor[0]) > 0:
                        token_mean = tweet_tensor[0].mean(dim=0)
                        total_each_person_tweets += token_mean
                    else:
                        total_each_person_tweets += torch.zeros(768)
                except Exception:
                    total_each_person_tweets += torch.zeros(768)
            count += 1

        # avoid division by zero
        if count > 0:
            total_each_person_tweets /= count
        tweets_list.append(total_each_person_tweets)

    # fallback safety
    if len(tweets_list) == 0:
        tweets_list = [torch.zeros(768)]

    tweet_tensor = torch.stack(tweets_list, 0)
    torch.save(tweet_tensor, path)
    print('Finished tweets embeddings.')
    return tweet_tensor

# ---------------- Run embeddings ----------------
des_tensor = Des_embbeding()
tweet_tensor = tweets_embedding()
print("All embeddings generated successfully!")
