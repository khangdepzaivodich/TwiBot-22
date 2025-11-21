import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os

# ---------------- Load merged dataset ----------------
user, tweet = fast_merge(dataset="Twibot-20")

# Extract user descriptions safely
user_text = [profile.get('description', None) if isinstance(profile, dict) else None 
             for profile in user['profile']]

# Extract tweet text safely
tweet_text = []
for t in tweet['tweet']:
    if isinstance(t, str):
        tweet_text.append(t)
    else:
        tweet_text.append('')  # Replace missing with empty string

# Load each_user_tweets from file
each_user_tweets = np.load('./processed_data/each_user_tweets.npy', allow_pickle=True)
# Convert dictionary to list of lists
each_user_tweets_list = [each_user_tweets[i] if i in each_user_tweets else [] for i in range(len(user))]

# ---------------- Feature extraction pipeline ----------------
feature_extract = pipeline(
    'feature-extraction',
    model='roberta-base',
    tokenizer='roberta-base',
    device=3,  # change to 0 if using first GPU
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
            if not each or str(each).strip() == '':
                des_vec.append(torch.zeros(768))
            else:
                feature = torch.Tensor(feature_extract(each))
                feature_tensor = feature[0][0]
                for tensor in feature[0][1:]:
                    feature_tensor += tensor
                feature_tensor /= feature.shape[1]
                des_vec.append(feature_tensor)
        if len(des_vec) == 0:  # fallback
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
        for user_tweets in tqdm(each_user_tweets_list):
            if not user_tweets:
                tweets_list.append(torch.zeros(768))
            else:
                total_each_person_tweets = None
                for j, tweet_idx in enumerate(user_tweets):
                    if j == 20:
                        break  # max 20 tweets
                    each_tweet = tweet_text[tweet_idx] if tweet_idx < len(tweet_text) else ''
                    if not each_tweet or str(each_tweet).strip() == '':
                        total_word_tensor = torch.zeros(768)
                    else:
                        each_tweet_tensor = torch.tensor(feature_extract(each_tweet))
                        total_word_tensor = each_tweet_tensor[0][0]
                        for tensor in each_tweet_tensor[0][1:]:
                            total_word_tensor += tensor
                        total_word_tensor /= each_tweet_tensor.shape[1]

                    if total_each_person_tweets is None:
                        total_each_person_tweets = total_word_tensor
                    else:
                        total_each_person_tweets += total_word_tensor

                total_each_person_tweets /= min(len(user_tweets), 20)
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
