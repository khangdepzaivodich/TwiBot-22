from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import os
from torch_geometric.data import Data, HeteroData
import json
import hashlib

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

# -------------------------
# Utilities
# -------------------------
def _safe_get_profile_field(profile, field, default=None):
    if isinstance(profile, dict):
        v = profile.get(field, default)
        # strip strings that include trailing spaces in original dataset
        if isinstance(v, str):
            return v.strip()
        return v
    return default

def _to_int_safe(x, default=0):
    try:
        if x is None:
            return default
        # remove stray spaces
        if isinstance(x, str):
            x = x.strip()
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def _deterministic_split_for_id(uid, ratios=(0.8, 0.1, 0.1)):
    """
    Deterministic split assignment from uid string.
    ratios is (train, val, test) and must sum to 1.0 approx.
    """
    # convert uid to a stable integer via hash
    h = int(hashlib.md5(str(uid).encode()).hexdigest()[:8], 16)
    p = (h % 10000) / 10000.0
    t, v, te = ratios
    if p < t:
        return "train"
    elif p < t + v:
        return "val"
    else:
        return "test"

# -------------------------
# Parsing helpers
# -------------------------
def _is_mixed_node_format(node_json_list):
    """
    Determine if node.json is the typical TwiBot mixed node format
    where IDs are like 'u...' and 't...' or there's explicit type markers.
    """
    if not node_json_list:
        return False
    # check first few entries
    for item in node_json_list[:10]:
        if not isinstance(item, dict):
            continue
        # typical TwiBot entries often have key "id" (lowercase) and sometimes prefixed 'u' or 't'
        idv = item.get("id") or item.get("ID")
        if isinstance(idv, str) and (idv.startswith("u") or idv.startswith("t")):
            return True
    return False

# -------------------------
# Main API
# -------------------------
def merge(dataset="Twibot-20", server_id="209"):
    """
    Backwards-compatible merge used in some downstream code.
    This will attempt to read node.json then merge with label.csv if available.
    For user-only JSON (your provided format), it will return a user dataframe.
    """
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset

    node_json_path = dataset_dir / "node.json"
    label_csv_path = dataset_dir / "label.csv"

    if not node_json_path.exists():
        raise FileNotFoundError(f"node.json not found at {node_json_path}")

    # Try to read node.json robustly
    try:
        # Check if file is line-delimited JSON or a JSON array
        with open(node_json_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if raw.startswith("["):
                node_list = json.loads(raw)
            else:
                # try line-delimited json: each line is a json object
                f.seek(0)
                node_list = [json.loads(line) for line in f if line.strip()]
    except Exception:
        # fallback to pandas (less flexible)
        node_df = pd.read_json(node_json_path, lines=True)
        # try to normalize to dataframe with 'id' and description fields
        if 'ID' in node_df.columns and 'profile' in node_df.columns:
            node_df['id'] = node_df['ID'].astype(str)
        return node_df

    # If it's the mixed TwiBot-style format, use the old behavior
    if _is_mixed_node_format(node_list):
        # convert to pandas quickly (pandas can handle list of dicts)
        df = pd.DataFrame(node_list)
        # ensure lowercase id column exists
        if "id" not in df.columns and "ID" in df.columns:
            df = df.rename(columns={"ID": "id"})
        df['id'] = df['id'].astype(str)
        # if label.csv exists, merge it in
        if label_csv_path.exists():
            label = pd.read_csv(label_csv_path)
            label['id'] = label['id'].astype(str)
            df = pd.merge(df, label, on='id', how='left')
        return df

    # Else assume user-only format like the sample you gave
    # Build user DataFrame and tweet DataFrame
    users = []
    tweets = []   # will be list of dicts with fields id,text,owner_id (owner user id)
    tweet_counter = 0

    for item in node_list:
        # user id
        uid = item.get("ID") or item.get("id") or item.get("Id")
        if uid is None:
            # skip malformed entry
            continue
        uid = str(uid).strip()

        profile = item.get("profile", {}) if isinstance(item.get("profile", {}), dict) else {}
        description = _safe_get_profile_field(profile, "description", None)
        username = _safe_get_profile_field(profile, "screen_name", None) or _safe_get_profile_field(profile, "name", None)
        # assemble a public_metrics dict similar to TwiBot original
        public_metrics = {
            "followers_count": _to_int_safe(_safe_get_profile_field(profile, "followers_count", None), 0),
            "following_count": _to_int_safe(_safe_get_profile_field(profile, "friends_count", None), 0),
            "listed_count": _to_int_safe(_safe_get_profile_field(profile, "listed_count", None), 0),
            "statuses_count": _to_int_safe(_safe_get_profile_field(profile, "statuses_count", None), 0)
        }
        # label: prefer label.csv if exists, else label in json item, else None
        label_val = None
        if label_csv_path.exists():
            try:
                label_df = pd.read_csv(label_csv_path)
                label_map = {str(x): y for x, y in zip(label_df['id'].astype(str), label_df['label'])}
                label_val = label_map.get(uid, None)
            except Exception:
                label_val = None
        if label_val is None:
            # fallback to 'label' inside json
            label_val = item.get("label") or item.get("label_id") or None
            if isinstance(label_val, str) and label_val.isdigit():
                # sometimes label is '0'/'1' as strings
                if label_val == "0":
                    label_val = "human"
                elif label_val == "1":
                    label_val = "bot"
        # split: prefer split.csv if exists else deterministic hash split
        split_val = None
        split_csv_path = dataset_dir / "split.csv"
        if split_csv_path.exists():
            try:
                split_df = pd.read_csv(split_csv_path)
                split_map = {str(x): y for x, y in zip(split_df['id'].astype(str), split_df['split'])}
                split_val = split_map.get(uid, None)
            except Exception:
                split_val = None
        if split_val is None:
            split_val = _deterministic_split_for_id(uid)

        users.append({
            "id": uid,
            "profile": profile,
            "description": description,
            "username": username,
            "public_metrics": public_metrics,
            "label": label_val,
            "split": split_val,
            # keep raw tweet list for convenience
            "tweets": item.get("tweet", [])
        })

        # extract tweets: each tweet gets a synthetic tweet id 't{tweet_counter}'
        item_tweets = item.get("tweet", []) or []
        for t_text in item_tweets:
            tid = f"t{tweet_counter}"
            tweets.append({
                "id": tid,
                "text": t_text,
                "owner_id": uid
            })
            tweet_counter += 1

    user_df = pd.DataFrame(users)
    tweet_df = pd.DataFrame(tweets)

    # ensure id columns are strings
    user_df['id'] = user_df['id'].astype(str)
    if not tweet_df.empty:
        tweet_df['id'] = tweet_df['id'].astype(str)

    return user_df, tweet_df

def split_user_and_tweet(df):
    """
    Deprecated / helper: attempts to split mixed node DataFrame into user,tweet dataframes.
    - If df.id contains 'u' / 't' prefix it uses that.
    - Else if df has 'profile' and 'tweet' columns and looks like user-only format,
      it returns (df, tweet_df) where tweet_df is exploded from df['tweet'] with synthetic tweet ids.
    """
    # ensure 'id' column exists
    if 'id' not in df.columns and 'ID' in df.columns:
        df = df.rename(columns={"ID": "id"})

    df['id'] = df['id'].astype(str)

    # Case A: TwiBot mixed format (id startswith u/t)
    if df['id'].str.startswith('u').any() or df['id'].str.startswith('t').any():
        users = df[df['id'].str.contains("^u")]
        tweets = df[df['id'].str.contains("^t")]
        return users.reset_index(drop=True), tweets.reset_index(drop=True)

    # Case B: user-only format (each row is a user with 'tweet' list)
    if 'tweet' in df.columns or 'profile' in df.columns:
        users = df.copy()
        # build tweets DataFrame by exploding the tweet lists and giving synthetic ids
        tweets_rows = []
        counter = 0
        for _, row in users.iterrows():
            uid = str(row.get('ID') or row.get('id'))
            tlist = row.get('tweet') or []
            print(row, tlist)
            # if tweet is a single string, convert to list
            if isinstance(tlist, str):
                tlist = [tlist]
            for t in (tlist or []):
                tweets_rows.append({"id": f"t{counter}", "text": t, "owner_id": uid})
                counter += 1
        tweets = pd.DataFrame(tweets_rows)
        # ensure consistent columns on users
        if 'description' not in users.columns:
            # try extract from profile
            users['description'] = users['profile'].apply(lambda p: p.get('description') if isinstance(p, dict) else None)
        users = users.reset_index(drop=True)
        tweets = tweets.reset_index(drop=True)
        return users, tweets

    # default fallback: return original df and empty tweets
    return df, pd.DataFrame(columns=["id", "text", "owner_id"])


def fast_merge(dataset="Twibot-20", server_id="209"):
    """
    Robust fast_merge that returns (user_df, tweet_df).
    - Supports original mixed TwiBot format (u/t ids).
    - Supports user-only JSON format (your sample). In that case tweets are extracted and assigned synthetic ids t0,t1,...
    - Merges label.csv and split.csv if they exist and apply to user_df.
    """
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    node_json_path = dataset_dir / "node.json"

    if not node_json_path.exists():
        raise FileNotFoundError(f"node.json not found at {node_json_path}")

    # Use merge() to do the heavy lifting
    merged = merge(dataset, server_id)

    # If merge returned a DataFrame (mixed format) then split
    if isinstance(merged, pd.DataFrame):
        user_df, tweet_df = split_user_and_tweet(merged)
    else:
        # merge returned (user_df, tweet_df)
        user_df, tweet_df = merged

    # ensure id types and useful columns exist
    if 'id' in user_df.columns:
        user_df['id'] = user_df['id'].astype(str)
    else:
        if 'ID' in user_df.columns:
            user_df['id'] = user_df['ID'].astype(str)

    # normalize description and profile columns
    if 'profile' in user_df.columns:
        user_df['description'] = user_df.get('description', user_df['profile'].apply(lambda p: p.get('description') if isinstance(p, dict) else None))
        user_df['username'] = user_df.get('username', user_df['profile'].apply(lambda p: p.get('screen_name') if isinstance(p, dict) else None))
        # build a public_metrics dict if not present
        if 'public_metrics' not in user_df.columns:
            def _pm(p):
                if isinstance(p, dict):
                    return {
                        "followers_count": _to_int_safe(p.get('followers_count', None), 0),
                        "following_count": _to_int_safe(p.get('friends_count', None), 0),
                        "listed_count": _to_int_safe(p.get('listed_count', None), 0),
                        "statuses_count": _to_int_safe(p.get('statuses_count', None), 0)
                    }
                return {"followers_count": 0, "following_count": 0, "listed_count": 0, "statuses_count": 0}
            user_df['public_metrics'] = user_df['profile'].apply(_pm)

    # if label.csv exists, merge in missing labels
    label_csv = dataset_dir / "label.csv"
    if label_csv.exists():
        try:
            label_df = pd.read_csv(label_csv)
            label_df['id'] = label_df['id'].astype(str)
            # merge only where user_df lacks label
            if 'label' not in user_df.columns:
                user_df = user_df.merge(label_df[['id', 'label']], on='id', how='left')
            else:
                # fillna from label_df
                label_map = dict(zip(label_df['id'].astype(str), label_df['label']))
                user_df['label'] = user_df.apply(lambda r: r['label'] if pd.notna(r['label']) else label_map.get(str(r['id'])), axis=1)
        except Exception:
            pass

    # if split.csv exists, attach split column; else create deterministic split
    split_csv = dataset_dir / "split.csv"
    if split_csv.exists():
        try:
            split_df = pd.read_csv(split_csv)
            split_map = dict(zip(split_df['id'].astype(str), split_df['split']))
            user_df['split'] = user_df.apply(lambda r: split_map.get(str(r['id']), r.get('split', _deterministic_split_for_id(r['id']))), axis=1)
        except Exception:
            user_df['split'] = user_df.get('split', user_df['id'].apply(_deterministic_split_for_id))
    else:
        user_df['split'] = user_df.get('split', user_df['id'].apply(_deterministic_split_for_id))

    # for tweet_df ensure minimal columns
    if tweet_df is None or tweet_df.empty:
        tweet_df = pd.DataFrame(columns=["id", "text", "owner_id"])
    else:
        if 'text' not in tweet_df.columns:
            # try column 'tweet' or 'tweet_text'
            if 'tweet' in tweet_df.columns:
                tweet_df = tweet_df.rename(columns={'tweet': 'text'})
            elif 'tweet_text' in tweet_df.columns:
                tweet_df = tweet_df.rename(columns={'tweet_text': 'text'})
        if 'id' in tweet_df.columns:
            tweet_df['id'] = tweet_df['id'].astype(str)

    # return user,tweet
    return user_df.reset_index(drop=True), tweet_df.reset_index(drop=True)

def merge_and_split(dataset="botometer-feedback-2019", server_id="209"):
    """
    Old convenience function kept for compatibility.
    """
    dataset_dir = get_data_dir(server_id) / dataset
    user_df, tweet_df = fast_merge(dataset, server_id)
    # If user_df already has split column with values train/val/test, return slices
    train = user_df[user_df["split"] == "train"]
    valid = user_df[user_df["split"] == "val"]
    test = user_df[user_df["split"] == "test"]
    return train, valid, test

@torch.no_grad()
def simple_vectorize(data):
    """
    Expects a pandas DataFrame with a 'description' column.
    Returns (feats_numpy, labels_numpy).
    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    # safe access
    descriptions = list(data.get("description", pd.Series([None]*len(data))))
    labels = list(data.get("label", pd.Series([None]*len(data))))
    labels = np.array([0 if (x == "human" or str(x) == "0") else 1 for x in labels], dtype=np.int32)

    feats = []
    for text in tqdm(descriptions):
        if text is None or str(text).strip() == "":
            feats.append(torch.randn(768))
            continue
        encoded_input = tokenizer(str(text), return_tensors='pt', truncation=True, padding=True, max_length=128)
        feats.append(model(**encoded_input)["pooler_output"][0])
    feats = torch.stack(feats, dim=0)
    return feats.numpy(), labels

@torch.no_grad()
def homo_graph_vectorize_only_user(include_node_feature=False, dataset="cresci-2015", server_id="209"):
    """
    Build a homogeneous user-user graph (edges from edge.csv) and optional features.
    """
    dataset_dir = get_data_dir(server_id) / dataset
    user, _ = fast_merge(dataset, server_id)
    user['id'] = user['id'].astype(str)

    # labels: convert whatever label format to 1:human,0:bot convention used earlier
    labels = torch.LongTensor([1 if (str(x) == "human" or str(x) == "0") else 0 for x in user.get('label', [None]*len(user))])
    unique_uid = set(user.id)
    user_index_to_uid = list(user.id)
    uid_to_user_index = {x: i for i, x in enumerate(user_index_to_uid)}
    user_text = [str(t) if t is not None else "" for t in user.get('description', [""]*len(user))]

    edge_path = dataset_dir / "edge.csv"
    if not edge_path.exists():
        # return empty graph structure if no edge file
        return Data(), uid_to_user_index, {}, user_index_to_uid

    edge = pd.read_csv(edge_path)
    # ensure string ids
    if 'source_id' in edge.columns:
        edge['source_id'] = edge['source_id'].astype(str)
    if 'target_id' in edge.columns:
        edge['target_id'] = edge['target_id'].astype(str)

    src, dst = [], []
    for s, t in zip(edge["source_id"], edge["target_id"]):
        if s in unique_uid and t in unique_uid:
            src.append(uid_to_user_index[s])
            dst.append(uid_to_user_index[t])
    edge_index = torch.LongTensor([src, dst])

    train_uid_with_label = user[user.split == "train"][["id", "split", "label"]] if "split" in user.columns else pd.DataFrame()
    valid_uid_with_label = user[user.split == "val"][["id", "split", "label"]] if "split" in user.columns else pd.DataFrame()
    test_uid_with_label = user[user.split == "test"][["id", "split", "label"]] if "split" in user.columns else pd.DataFrame()

    if include_node_feature:
        if "user_info.pt" not in os.listdir(dataset_dir):
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained("roberta-base")
            user_text_feats = []
            for t in tqdm(user_text):
                encoded_input = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=128)
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
            vals = tuple(torch.load(dataset_dir / "user_info.pt").values())
            return vals
    else:
        return user_text, edge_index, labels, uid_to_user_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label

def df_to_mask(uid_with_label, uid_to_user_index, phase="train"):
    """
    uid_with_label: a dataframe with column 'id' and 'split'
    uid_to_user_index: mapping id->index
    """
    user_list = list(uid_with_label[uid_with_label.split == phase].id)
    phase_index = list(map(lambda x: uid_to_user_index[str(x)], user_list))
    return torch.LongTensor(phase_index)

## debug
if __name__ == "__main__":
    # quick smoke test when run directly (will raise if dataset not present)
    try:
        u, t = fast_merge("Twibot-20", "209")
        print("users:", len(u), "tweets:", len(t))
    except Exception as e:
        print("fast_merge failed:", e)
