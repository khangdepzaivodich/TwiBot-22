import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os

# ---------------- Load dataset ----------------
user, tweet = fast_merge(dataset="Twibot-20")

# Extract user descriptions from the nested profile
user_text = [
    profile.get('description', None) if isinstance(profile, dict) else None
    for profile in user['profile']
]

# Extract tweets
tweet_text = []
for t in tweet['tweet']:
    if isinstance(t, str):
        tweet_text.append(t)
    elif isinstance(t, list) and len(t) > 0:
        tweet_text.append(str(t[0]))
    else:
        tweet_text.append(None)

# Load each_user_tweets mapping
each_user_tweets = np.load('./processed_data/each_user_tweets.npy', allow_pickle=True).item()

# Convert each_user_tweets dict to list of lists
max_uid = len(user)
tweets_per_user = [[] for _ in range(max_uid)]
for uid, t_list in each_user_tweets.items():
    if isinstance(t_list, (list, np.ndarray)):
        tweets_per_user[uid] = list(t_list)
    else:
        tweets_per_user[uid] = []

# ---------------- Feature extraction pipeline ----------------
feature_extract = pipeline(
    'feature-extraction',
    model='roberta-base',
    tokenizer='roberta-base',
    device=0,   # Change to 0 if using GPU, -1 for CPU
    padding=True,
    truncation=True,
    max_length=50,
    add_special_tokens=True
)

# ---------------- User description embeddings ----------------
def Des_embbeding():
    print('Running user description embeddings')
    path = "./processed_data/des_tensor.pt"
    if not os.path.exists(path):
        des_vec = []
        for each in tqdm(user_text):
            try:
                if not each or str(each).strip() == '':
                    des_vec.append(torch.zeros(768))
                else:
                    feature = torch.Tensor(feature_extract(each))
                    if feature and len(feature) > 0 and len(feature[0]) > 0:
                        feature_tensor = feature[0][0]
                        for tensor in feature[0][1:]:
                            feature_tensor += tensor
                        feature_tensor /= feature.shape[1]
                        des_vec.append(feature_tensor)
                    else:
                        des_vec.append(torch.zeros(768))
            except Exception:
                des_vec.append(torch.zeros(768))  # fallback
        # Ensure we have the right length
        if len(des_vec) != len(user_text):
            des_vec = [torch.zeros(768) for _ in range(len(user_text))]
        des_tensor = torch.stack(des_vec, 0)
        torch.save(des_tensor, path)
    else:
        des_tensor = torch.load(path)
    print('Finished user description embeddings')
    return des_tensor

# ---------------- Tweets embeddings ----------------
def tweets_embedding():
    print('Running tweets embeddings')
    path = "./processed_data/tweets_tensor.pt"
    if not os.path.exists(path):
        tweets_list = []
        for i, user_tweet_ids in enumerate(tqdm(tweets_per_user)):
            if len(user_tweet_ids) == 0:
                tweets_list.append(torch.zeros(768))
            else:
                total_each_person_tweets = None
                for j, tid in enumerate(user_tweet_ids):
                    if j == 20:  # Max 20 tweets per user
                        break
                    each_tweet = tweet_text[tid] if tid < len(tweet_text) else None
                    try:
                        if not each_tweet:
                            total_word_tensor = torch.zeros(768)
                        else:
                            feature = torch.Tensor(feature_extract(each_tweet))
                            if feature and len(feature) > 0 and len(feature[0]) > 0:
                                total_word_tensor = feature[0][0]
                                for tensor in feature[0][1:]:
                                    total_word_tensor += tensor
                                total_word_tensor /= feature.shape[1]
                            else:
                                total_word_tensor = torch.zeros(768)
                    except Exception:
                        total_word_tensor = torch.zeros(768)

                    if total_each_person_tweets is None:
                        total_each_person_tweets = total_word_tensor
                    else:
                        total_each_person_tweets += total_word_tensor

                total_each_person_tweets /= min(len(user_tweet_ids), 20)
                tweets_list.append(total_each_person_tweets)

        tweet_tensor = torch.stack(tweets_list)
        torch.save(tweet_tensor, path)
    else:
        tweet_tensor = torch.load(path)
    print('Finished tweets embeddings')
    return tweet_tensor

# ---------------- Run embeddings ----------------
Des_embbeding()
tweets_embedding()
