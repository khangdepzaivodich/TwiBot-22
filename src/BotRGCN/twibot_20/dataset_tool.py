from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import os
from torch_geometric.data import Data, HeteroData

def get_data_dir(server_id):
    if server_id == "206":
        return Path("/new_temp/fsb/Twibot22-baselines/datasets")
    elif server_id == "208":
        return Path("")
    elif server_id == "209":
        return Path("../datasets")
    else:
        raise NotImplementedError

dataset_names = [
    'botometer-feedback-2019', 'botwiki-2019', 'celebrity-2019', 'cresci-2015',
    'cresci-2017', 'cresci-rtbust-2019', 'cresci-stock-2018', 'gilani-2017',
    'midterm-2018', 'political-bots-2019', 'pronbots-2019', 'vendor-purchased-2019',
    'verified-2019', "Twibot-20"
]

def merge(dataset="Twibot-20", server_id="209"):
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    node_info = pd.read_csv(dataset_dir / "node.json")
    node_info['id'] = node_info['id'].astype(str)  # ensure string type
    label = pd.read_csv(dataset_dir / "label.csv")
    node_info = pd.merge(node_info, label)
    return node_info

def split_user_and_tweet(df):
    df['id'] = df['id'].astype(str)
    df = df[df['id'].str.len() > 0]
    return df[df['id'].str.contains("^u")], df[df['id'].str.contains("^t")]

def fast_merge(dataset="Twibot-20", server_id="209"):
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    node_info = pd.read_json(dataset_dir / "node.json")
    node_info['id'] = node_info['id'].astype(str)
    label = pd.read_csv(dataset_dir / "label.csv")
    split = pd.read_csv(dataset_dir / "split.csv")
    
    user, tweet = split_user_and_tweet(node_info)
    
    id_to_label = {str(label['id'][i]): label['label'][i] for i in range(len(label))}
    id_to_split = {str(split['id'][i]): split['split'][i] for i in range(len(split))}
    
    user["label"] = "None"
    user["split"] = "None"
    
    for i in tqdm(range(len(user))):
        uid = str(user["id"][i])
        if uid in id_to_label:
            user.at[i, "label"] = id_to_label[uid]
            user.at[i, "split"] = id_to_split[uid]
    
    return user, tweet

def merge_and_split(dataset="botometer-feedback-2019", server_id="209"):
    dataset_dir = get_data_dir(server_id) / dataset
    node_info = pd.read_json(dataset_dir / "node.json")
    node_info['id'] = node_info['id'].astype(str)
    label = pd.read_csv(dataset_dir / "label.csv")
    split = pd.read_csv(dataset_dir / "split.csv")
    node_info = pd.merge(node_info, label)
    node_info = pd.merge(node_info, split)
    
    train = node_info[node_info["split"] == "train"]
    valid = node_info[node_info["split"] == "val"]
    test = node_info[node_info["split"] == "test"]
    return train, valid, test

@torch.no_grad()
def simple_vectorize(data):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    descriptions = list(data["description"])
    labels = list(data["label"])
    labels = np.array([0 if x == "human" else 1 for x in labels], dtype=np.int32)
    
    feats = []
    for text in tqdm(descriptions):
        if text is None:
            feats.append(torch.randn(768))
            continue
        encoded_input = tokenizer(text, return_tensors='pt')
        feats.append(model(**encoded_input)["pooler_output"][0])
    feats = torch.stack(feats, dim=0)
    return feats.numpy(), labels

@torch.no_grad()
def homo_graph_vectorize_only_user(include_node_feature=False, dataset="cresci-2015", server_id="209"):
    dataset_dir = get_data_dir(server_id) / dataset
    user, _ = fast_merge(dataset, server_id)
    user['id'] = user['id'].astype(str)
    
    labels = torch.LongTensor([1 if x == "human" else 0 for x in user.label])
    unique_uid = set(user.id)
    user_index_to_uid = list(user.id)
    uid_to_user_index = {x: i for i, x in enumerate(user_index_to_uid)}
    user_text = [str(t) if t is not None else "" for t in user.description]
    
    edge = pd.read_csv(dataset_dir / "edge.csv")
    edge['source_id'] = edge['source_id'].astype(str)
    edge['target_id'] = edge['target_id'].astype(str)
    
    src, dst = [], []
    for s, t in zip(edge["source_id"], edge["target_id"]):
        if s in unique_uid and t in unique_uid:
            src.append(uid_to_user_index[s])
            dst.append(uid_to_user_index[t])
    edge_index = torch.LongTensor([src, dst])
    
    train_uid_with_label = user[user.split == "train"][["id", "split", "label"]]
    valid_uid_with_label = user[user.split == "val"][["id", "split", "label"]]
    test_uid_with_label = user[user.split == "test"][["id", "split", "label"]]
    
    if include_node_feature:
        if "user_info.pt" not in os.listdir(dataset_dir):
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained("roberta-base")
            user_text_feats = []
            for t in tqdm(user_text):
                encoded_input = tokenizer(t, return_tensors="pt")
                user_text_feats.append(model(**encoded_input)["pooler_output"][0])
            user_text_feats = torch.stack(user_text_feats, dim=0)
            user_info = {
                "user_text_feats": user_text_feats,
                "edge_index": edge_index,
                "labels": labels,
                "uid_to_user_index": uid_to_user_index,
                "train_uid_with_label": train_uid_with_label,
                "valid_uid_with_label": valid_uid_with_label,
                "test_uid_with_label": test_uid_with_label,
            }
            torch.save(user_info, dataset_dir / "user_info.pt")
            return user_text_feats, edge_index, labels, uid_to_user_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label
        else:
            return tuple(torch.load(dataset_dir / "user_info.pt").values())
    else:
        return user_text, edge_index, labels, uid_to_user_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label

def df_to_mask(uid_with_label, uid_to_user_index, phase="train"):
    user_list = list(uid_with_label[uid_with_label.split == phase].id)
    phase_index = list(map(lambda x: uid_to_user_index[str(x)], user_list))
    return torch.LongTensor(phase_index)

## debug
if __name__ == "__main__":
    homo_graph_vectorize_only_user(True)
