import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
from dataset_tool import fast_merge, df_to_mask
import os

print('Loading raw data')
edge = pd.read_csv("../datasets/Twibot-20/edge.csv")
label = pd.read_csv("../datasets/Twibot-20/label.csv")
split = pd.read_csv("../datasets/Twibot-20/split.csv")

print('Processing raw data')
user, tweet = fast_merge(dataset='Twibot-20')
path = 'processed_data/'

if not os.path.exists(path):
    os.mkdir(path)

# ---------------- Labels ----------------
print('Extracting labels and splits')
label_list = label['label']
label_tensor = torch.tensor(label_list, dtype=torch.long)
torch.save(label_tensor, os.path.join(path,'label.pt'))

train_uid_with_label = user[user.split == "train"][["id", "split", "label"]]
valid_uid_with_label = user[user.split == "val"][["id", "split", "label"]]
test_uid_with_label = user[user.split == "test"][["id", "split", "label"]]

user_index_to_uid = list(user.id)
tweet_index_to_tid = list(tweet.id)
uid_to_user_index = {x: i for i, x in enumerate(user_index_to_uid)}
tid_to_tweet_index = {x: i for i, x in enumerate(tweet_index_to_tid)}

train_mask = df_to_mask(train_uid_with_label, uid_to_user_index, "train")
valid_mask = df_to_mask(valid_uid_with_label, uid_to_user_index, "val")
test_mask = df_to_mask(test_uid_with_label, uid_to_user_index, "test")
torch.save(train_mask, os.path.join(path,"train_idx.pt"))
torch.save(valid_mask, os.path.join(path,"val_idx.pt"))
torch.save(test_mask, os.path.join(path,"test_idx.pt"))

# ---------------- Graph ----------------
print('Extracting graph info')
edge_index = []
edge_type = []
for i in tqdm(range(len(edge))):
    if edge['relation'][i] == 'post':
        continue
    try:
        source_id = uid_to_user_index[edge['source_id'][i]]
        target_id = uid_to_user_index[edge['target_id'][i]]
    except KeyError:
        continue
    else:
        edge_index.append([source_id, target_id])
        edge_type.append(0 if edge['relation'][i] == 'friend' else 1)

torch.save(torch.tensor(edge_index, dtype=torch.long).t(), os.path.join(path,'edge_index.pt'))
torch.save(torch.tensor(edge_type, dtype=torch.long), os.path.join(path,'edge_type.pt'))

# ---------------- Numeric properties ----------------
print('Extracting num_properties')
following_count, statuses, followers_count, screen_name_length, created_at_list = [], [], [], [], []

for profile in user['profile']:
    following_count.append(int(profile.get('friends_count', 0)))
    statuses.append(int(profile.get('listed_count', 0)))
    followers_count.append(int(profile.get('followers_count', 0)))
    screen_name_length.append(len(profile.get('screen_name', '')) if profile.get('screen_name') else 0)
    created_at_list.append(profile.get('created_at', None))

# Normalize numeric features
followers_count = torch.tensor((np.array(followers_count)-np.mean(followers_count))/np.std(followers_count), dtype=torch.float32)
following_count = torch.tensor((np.array(following_count)-np.mean(following_count))/np.std(following_count), dtype=torch.float32)
statuses = torch.tensor((np.array(statuses)-np.mean(statuses))/np.std(statuses), dtype=torch.float32)
screen_name_length = torch.tensor((np.array(screen_name_length)-np.mean(screen_name_length))/np.std(screen_name_length), dtype=torch.float32)

# Active days
date0 = dt.strptime('Tue Sep 1 00:00:00 +0000 2020 ', '%a %b %d %X %z %Y ')
active_days = []
for dt_str in created_at_list:
    if dt_str:
        dt_obj = dt.strptime(dt_str.strip(), '%a %b %d %X %z %Y')
        active_days.append((date0 - dt_obj).days)
    else:
        active_days.append(0)
active_days = torch.tensor((np.array(active_days)-np.mean(active_days))/np.std(active_days), dtype=torch.float32)

num_properties_tensor = torch.cat([
    followers_count.unsqueeze(1),
    active_days.unsqueeze(1),
    screen_name_length.unsqueeze(1),
    following_count.unsqueeze(1),
    statuses.unsqueeze(1)
], dim=1)
torch.save(num_properties_tensor, os.path.join(path,'num_properties_tensor.pt'))

# ---------------- Categorical properties ----------------
print('Extracting cat_properties')
default_profile_image, protected_list, verified_list = [], [], []

for profile in user['profile']:
    default_profile_image.append(1 if profile.get('default_profile_image') == 'True' else 0)
    protected_list.append(1 if profile.get('protected') == 'True' else 0)
    verified_list.append(1 if profile.get('verified') == 'True' else 0)

cat_properties_tensor = torch.tensor(default_profile_image, dtype=torch.float).reshape(-1,1)
torch.save(cat_properties_tensor, os.path.join(path,'cat_properties_tensor.pt'))

# ---------------- Each user tweets ----------------
print('Extracting each_user_tweets')
dict_user_tweets = {}

for i, u in user.iterrows():
    tweets = u.get("tweets", [])
    if not isinstance(tweets, list):
        tweets = []
    dict_user_tweets[i] = tweets

torch.save(dict_user_tweets, os.path.join(path, 'each_user_tweets.pt'))

print("Preprocessing complete!")
