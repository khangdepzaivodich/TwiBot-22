import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os

# ---------------- Load merged dataset ----------------
user, tweet = fast_merge(dataset="Twibot-20")

# Extract user descriptions from the nested profile
user_text = [
    profile.get('description', None) if isinstance(profile, dict) else None
    for profile in user['profile']
]

# Extract tweets
tweet_text = [text for text in tweet['tweet']]

# Load preprocessed tweets per user (.npy, not .pt)
each_user_tweets = np.load('./processed_data/each_user_tweets.npy', allow_pickle=True)
each_user_tweets = [list(tweets) for tweets in each_user_tweets]

# ---------------- Feature extraction pipeline ----------------
feature_extract = pipeline(
    'feature-extraction',
    model='roberta-base',
    tokenizer='roberta-base',
    device=3,  # change to your CUDA device ID
    padding=True,
    truncation=True,
    max_length=50,
    add_special_tokens=True
)

# ---------------- User description embeddings ----------------
def Des_embbeding():
    print('Running feature1 embedding')
    path = "./processed_data/des_tensor.pt"
    if not os.path.exists(path):
        des_vec = []
        for each in tqdm(user_text):
            if each is None:
                des_vec.append(torch.zeros(768))
            else:
                feature = torch.Tensor(feature_extract(each))
                # Average word embeddings
                feature_tensor = feature[0][0]
                for tensor in feature[0][1:]:
                    feature_tensor += tensor
                feature_tensor /= feature.shape[1]
                des_vec.append(feature_tensor)

        des_tensor = torch.stack(des_vec, 0)
        torch.save(des_tensor, path)
    else:
        des_tensor = torch.load(path)
    print('Finished')
    return des_tensor

# ---------------- Tweets embeddings ----------------
def tweets_embedding():
    print('Running feature2 embedding')
    path = "./processed_data/tweets_tensor.pt"
    if not os.path.exists(path):
        tweets_list = []
        for i in tqdm(range(len(each_user_tweets))):
            user_tweets = each_user_tweets[i]
            if len(user_tweets) == 0:
                total_each_person_tweets = torch.zeros(768)
            else:
                total_each_person_tweets = None
                for j, tweet_idx in enumerate(user_tweets):
                    if j == 20:  # Use up to 20 tweets per user
                        break
                    each_tweet = tweet_text[tweet_idx]
                    if each_tweet is None:
                        total_word_tensor = torch.zeros(768)
                    else:
                        each_tweet_tensor = torch.tensor(feature_extract(each_tweet))
                        # Average word embeddings
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
    print('Finished')
    return tweet_tensor

# ---------------- Run embeddings ----------------
Des_embbeding()
tweets_embedding()
