import torch
from tqdm import tqdm
from dataset_tool import fast_merge
from transformers import pipeline
import os

print('Loading dataset')
user, tweet = fast_merge(dataset="Twibot-20")

# Extract text
user_text = [profile.get('description', None) if isinstance(profile, dict) else None
             for profile in user['profile']]
tweet_text = [t for t in tweet['tweet']]  # tweet['tweet'] contains tweet content

# Load each_user_tweets
each_user_tweets_dict = torch.load('./processed_data/each_user_tweets.pt')
# Convert dict to list of lists
each_user_tweets = [each_user_tweets_dict[i] for i in range(len(each_user_tweets_dict))]

# Feature extraction
feature_extract = pipeline(
    'feature-extraction',
    model='roberta-base',
    tokenizer='roberta-base',
    device=0,  # set device (GPU index)
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
            if not each:
                des_vec.append(torch.zeros(768))
            else:
                feature = torch.Tensor(feature_extract(each))
                feature_tensor = feature[0][0]
                for tensor in feature[0][1:]:
                    feature_tensor += tensor
                feature_tensor /= feature.shape[1]
                des_vec.append(feature_tensor)
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
        tweets_vec = []
        for user_tweets in tqdm(each_user_tweets):
            if len(user_tweets) == 0:
                tweets_vec.append(torch.zeros(768))
                continue
            total_vec = None
            for j, idx in enumerate(user_tweets):
                if j == 20:
                    break
                t_text = tweet_text[idx]
                if not t_text:
                    t_vec = torch.zeros(768)
                else:
                    t_feat = torch.Tensor(feature_extract(t_text))
                    t_vec = t_feat[0][0]
                    for tensor in t_feat[0][1:]:
                        t_vec += tensor
                    t_vec /= t_feat.shape[1]
                if total_vec is None:
                    total_vec = t_vec
                else:
                    total_vec += t_vec
            total_vec /= min(len(user_tweets), 20)
            tweets_vec.append(total_vec)
        tweets_tensor = torch.stack(tweets_vec)
        torch.save(tweets_tensor, path)
    else:
        tweets_tensor = torch.load(path)
    print('Finished tweets embeddings')
    return tweets_tensor

# Run embeddings
Des_embbeding()
tweets_embedding()
