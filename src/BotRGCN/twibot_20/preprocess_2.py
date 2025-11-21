import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os

# ---------------- Load dataset ----------------
print("Loading dataset...")
user, tweet = fast_merge(dataset="Twibot-20")

# ---------------- Extract user descriptions ----------------
user_text = []
for profile in user['profile']:
    if isinstance(profile, dict):
        desc = profile.get('description', None)
        if desc is not None and str(desc).strip() != '':
            user_text.append(desc)
        else:
            user_text.append(None)
    else:
        user_text.append(None)

# ---------------- Extract tweets text ----------------
tweet_text = []
for t in tweet['tweet']:
    if isinstance(t, str):
        tweet_text.append(t)
    else:
        tweet_text.append(None)

# ---------------- Load each_user_tweets ----------------
print("Loading per-user tweet indices...")
each_user_tweets_path = './processed_data/each_user_tweets.npy'
if os.path.exists(each_user_tweets_path):
    each_user_tweets_dict = np.load(each_user_tweets_path, allow_pickle=True).item()
    # Ensure a list for each user
    each_user_tweets = [list(each_user_tweets_dict.get(i, [])) for i in range(len(user))]
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
        print('Finished user description embeddings (loaded cached).')
        return des_tensor

    des_vec = []
    for each in tqdm(user_text):
        if each is None:
            des_vec.append(torch.zeros(768))
        else:
            try:
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
                des_vec.append(torch.zeros(768))

    # Ensure des_vec is never empty
    if len(des_vec) == 0:
        des_vec = [torch.zeros(768) for _ in range(len(user))]

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
        print('Finished tweets embeddings (loaded cached).')
        return tweet_tensor

    tweets_list = []
    for i in tqdm(range(len(each_user_tweets))):
        user_tweets = each_user_tweets[i]
        if len(user_tweets) == 0:
            total_each_person_tweets = torch.zeros(768)
        else:
            total_each_person_tweets = None
            for j, tweet_idx in enumerate(user_tweets):
                if j == 20:
                    break
                each_tweet = tweet_text[tweet_idx] if tweet_idx < len(tweet_text) else None
                if each_tweet is None:
                    total_word_tensor = torch.zeros(768)
                else:
                    try:
                        each_tweet_tensor = torch.tensor(feature_extract(each_tweet))
                        if each_tweet_tensor and len(each_tweet_tensor) > 0 and len(each_tweet_tensor[0]) > 0:
                            total_word_tensor = each_tweet_tensor[0][0]
                            for tensor in each_tweet_tensor[0][1:]:
                                total_word_tensor += tensor
                            total_word_tensor /= each_tweet_tensor.shape[1]
                        else:
                            total_word_tensor = torch.zeros(768)
                    except Exception:
                        total_word_tensor = torch.zeros(768)

                if total_each_person_tweets is None:
                    total_each_person_tweets = total_word_tensor
                else:
                    total_each_person_tweets += total_word_tensor

            total_each_person_tweets /= min(len(user_tweets), 20)

        tweets_list.append(total_each_person_tweets)

    # Ensure tweets_list is never empty
    if len(tweets_list) == 0:
        tweets_list = [torch.zeros(768) for _ in range(len(user))]

    tweet_tensor = torch.stack(tweets_list)
    torch.save(tweet_tensor, path)
    print('Finished tweets embeddings.')
    return tweet_tensor

# ---------------- Run embeddings ----------------
des_tensor = Des_embbeding()
tweet_tensor = tweets_embedding()
print("All embeddings processed successfully!")
