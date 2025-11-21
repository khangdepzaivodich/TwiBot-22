import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os

# ---------------- Load merged dataset ----------------
user, tweet = fast_merge(dataset="Twibot-20")

# ---------------- Extract user descriptions ----------------
user_text = []
for profile in user['profile']:
    if isinstance(profile, dict):
        user_text.append(profile.get('description', None))
    else:
        user_text.append(None)

# ---------------- Extract tweet texts ----------------
tweet_text = []
for t in tweet['tweet']:
    if isinstance(t, str):
        tweet_text.append(t)
    else:
        tweet_text.append(None)

# ---------------- Load preprocessed tweets per user ----------------
# each_user_tweets.npy was saved as np array, load with numpy
each_user_tweets = np.load('./processed_data/each_user_tweets.npy', allow_pickle=True)
each_user_tweets = [list(tweets) if isinstance(tweets, (list, np.ndarray)) else [] for tweets in each_user_tweets]

# ---------------- Feature extraction pipeline ----------------
feature_extract = pipeline(
    'feature-extraction',
    model='roberta-base',
    tokenizer='roberta-base',
    device=3,  # Change device if needed
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
        print('Loaded existing description embeddings.')
        return des_tensor

    des_vec = []
    for each in tqdm(user_text):
        if not each:
            des_vec.append(torch.zeros(768))
        else:
            try:
                feature = torch.Tensor(feature_extract(each))
                # Average word embeddings
                feature_tensor = feature[0][0]
                for tensor in feature[0][1:]:
                    feature_tensor += tensor
                feature_tensor /= feature.shape[1]
                des_vec.append(feature_tensor)
            except Exception as e:
                # Fallback in case feature extraction fails
                des_vec.append(torch.zeros(768))

    # Safety check
    if len(des_vec) == 0:
        des_vec.append(torch.zeros(768))

    des_tensor = torch.stack(des_vec, 0)
    torch.save(des_tensor, path)
    print('Finished user description embeddings.')
    return des_tensor

# ---------------- Tweets embeddings ----------------
def tweets_embedding():
    print('Running user tweets embeddings...')
    path = "./processed_data/tweets_tensor.pt"
    if os.path.exists(path):
        tweet_tensor = torch.load(path)
        print('Loaded existing tweets embeddings.')
        return tweet_tensor

    tweets_list = []
    for i, user_tweet_indices in enumerate(tqdm(each_user_tweets)):
        if len(user_tweet_indices) == 0:
            tweets_list.append(torch.zeros(768))
            continue

        total_each_person_tweets = None
        for j, tweet_idx in enumerate(user_tweet_indices):
            if j == 20:  # Max 20 tweets per user
                break

            each_tweet = tweet_text[tweet_idx] if tweet_idx < len(tweet_text) else None
            if not each_tweet:
                total_word_tensor = torch.zeros(768)
            else:
                try:
                    each_tweet_tensor = torch.tensor(feature_extract(each_tweet))
                    total_word_tensor = each_tweet_tensor[0][0]
                    for tensor in each_tweet_tensor[0][1:]:
                        total_word_tensor += tensor
                    total_word_tensor /= each_tweet_tensor.shape[1]
                except Exception as e:
                    total_word_tensor = torch.zeros(768)

            if total_each_person_tweets is None:
                total_each_person_tweets = total_word_tensor
            else:
                total_each_person_tweets += total_word_tensor

        total_each_person_tweets /= min(len(user_tweet_indices), 20)
        tweets_list.append(total_each_person_tweets)

    tweet_tensor = torch.stack(tweets_list)
    torch.save(tweet_tensor, path)
    print('Finished user tweets embeddings.')
    return tweet_tensor

# ---------------- Run embeddings ----------------
Des_embbeding()
tweets_embedding()
