# =========================================================================
# Copyright (C) 2024. FuxiCTR Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import polars as pl
import numpy as np
import os
from pandas.core.common import flatten
from datetime import datetime
from sklearn.decomposition import PCA
import gc


# Download the datasets and put them to the following folders
train_path = "./train/"
dev_path = "./validation/"
test_path = "./test/"
dataset_version = "Ebnerd_large_x1"
image_emb_path = "image_embeddings.parquet"
contrast_emb_path = "contrastive_vector.parquet"
MAX_SEQ_LEN = 50

print("Preprocess news info...")
train_news_file = os.path.join(train_path, "articles.parquet")
train_news = pl.scan_parquet(train_news_file)
test_news_file = os.path.join(test_path, "articles.parquet")
test_news = pl.scan_parquet(test_news_file)
news = pl.concat([train_news, test_news])
news = news.unique(subset=['article_id'])
news = news.fill_null("")

def map_feat_id_func(df, column, seq_feat=False):
    feat_set = set(flatten(df[column].to_list()))
    map_dict = dict(zip(list(feat_set), range(1, 1 + len(feat_set))))
    if seq_feat:
        df = df.with_columns(pl.col(column).apply(lambda x: [map_dict.get(i, 0) for i in x]))
    else:
        df = df.with_columns(pl.col(column).apply(lambda x: map_dict.get(x, 0)).cast(str))
    return df

def tokenize_seq(df, column, map_feat_id=True, max_seq_length=5, sep="^"):
    df = df.with_columns(pl.col(column).apply(lambda x: x[-max_seq_length:]))
    if map_feat_id:
        df = map_feat_id_func(df, column, seq_feat=True)
    df = df.with_columns(pl.col(column).apply(lambda x: f"{sep}".join(str(i) for i in x)))
    return df

news = news.select(['article_id', 'published_time', 'last_modified_time', 'premium',
                    'article_type', 'ner_clusters', 'topics', 'category', 'subcategory',
                    'total_inviews', 'total_pageviews', 'total_read_time',
                    'sentiment_score', 'sentiment_label'])
news = (
    news.with_columns(subcat1=pl.col('subcategory').apply(lambda x: str(x[0]) if len(x) > 0 else ""))
    .collect()
)
news2cat = dict(zip(news["article_id"].cast(str), news["category"].cast(str)))
news2subcat = dict(zip(news["article_id"].cast(str), news["subcat1"].cast(str)))
news = tokenize_seq(news, 'ner_clusters', map_feat_id=True)
news = tokenize_seq(news, 'topics', map_feat_id=True)
news = tokenize_seq(news, 'subcategory', map_feat_id=False)
news = map_feat_id_func(news, "sentiment_label")
news = map_feat_id_func(news, "article_type")
news2sentiment = dict(zip(news["article_id"].cast(str), news["sentiment_label"]))
news2type = dict(zip(news["article_id"].cast(str), news["article_type"]))
print(news.head())
print("Save news info...")
os.makedirs(dataset_version, exist_ok=True)
with open(f"./{dataset_version}/news_info.jsonl", "w") as f:
    f.write(news.write_json(row_oriented=True, pretty=True))

print("Preprocess behavior data...")

def join_data(data_path):
    history_file = os.path.join(data_path, "history.parquet")
    history_df = pl.scan_parquet(history_file)
    history_df = history_df.rename({"article_id_fixed": "hist_id", 
                                    "read_time_fixed": "hist_read_time",
                                    "impression_time_fixed": "hist_time",
                                    "scroll_percentage_fixed": "hist_scroll_percent"})
    history_df = tokenize_seq(history_df, 'hist_id', map_feat_id=False, max_seq_length=MAX_SEQ_LEN)
    # history_df["hist_time"] = history_df["hist_time"].map(
    #     lambda x: [datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%f") for v in x[-MAX_SEQ_LEN:]])
    history_df = history_df.select(["user_id", "hist_id"])
    history_df = history_df.with_columns(
        pl.col("hist_id").apply(lambda x: "^".join([news2cat.get(i, "") for i in x.split("^")])).alias("hist_cat"),
        pl.col("hist_id").apply(lambda x: "^".join([news2subcat.get(i, "") for i in x.split("^")])).alias("hist_subcat1"),
        pl.col("hist_id").apply(lambda x: "^".join([news2sentiment.get(i, "") for i in x.split("^")])).alias("hist_sentiment"),
        pl.col("hist_id").apply(lambda x: "^".join([news2type.get(i, "") for i in x.split("^")])).alias("hist_type")
    )
    history_df = history_df.collect()
    behavior_file = os.path.join(data_path, "behaviors.parquet")
    sample_df = pl.scan_parquet(behavior_file)
    if "test/" in data_path:
        sample_df = (
            sample_df.rename({"article_ids_inview": "article_id"})
            .explode('article_id')
        )
        sample_df = sample_df.with_columns(
            pl.lit(None).alias("trigger_id"),
            pl.lit(0).alias("click")
        )
    else:
        sample_df = (
            sample_df.rename({"article_id": "trigger_id"})
            .rename({"article_ids_inview": "article_id"})
            .explode('article_id')
            .with_columns(click=pl.col("article_id").is_in(pl.col("article_ids_clicked")).cast(pl.Int8))
            .drop(["article_ids_clicked"])
        )
    sample_df = (
        sample_df.collect()
        .join(news, on='article_id', how="left")
        .join(history_df, on='user_id', how="left")
        .with_columns(
            publish_days=(pl.col('impression_time') - pl.col('published_time')).dt.days().cast(pl.Int32),
            publish_hours=(pl.col('impression_time') - pl.col('published_time')).dt.hours().cast(pl.Int32),
            impression_hour=pl.col('impression_time').dt.hour().cast(pl.Int32),
            impression_weekday=pl.col('impression_time').dt.weekday().cast(pl.Int32)
        )
        .with_columns(
            pl.col("publish_days").clip_max(3).alias("pulish_3day"),
            pl.col("publish_days").clip_max(7).alias("pulish_7day"),
            pl.col("publish_days").clip_max(30),
            pl.col("publish_hours").clip_max(24)
        )
        .drop(["impression_time", "published_time", "last_modified_time"])
    )
    print(sample_df.columns)
    return sample_df

train_df = join_data(train_path)
print(train_df.head())
print("Train samples", train_df.shape)
train_df.write_csv(f"./{dataset_version}/train.csv")
del train_df

valid_df = join_data(dev_path)
print(valid_df.head())
print("Validation samples", valid_df.shape)
valid_df.write_csv(f"./{dataset_version}/valid.csv")
del valid_df
gc.collect()

test_df = join_data(test_path)
print(test_df.head())
print("Test samples", test_df.shape)
test_df.write_csv(f"./{dataset_version}/test.csv")
del test_df
gc.collect()

print("Preprocess pretrained embeddings...")
image_emb_df = pl.read_parquet(image_emb_path)
pca = PCA(n_components=64)
image_emb = pca.fit_transform(np.array(image_emb_df["image_embedding"].to_list()))
print("image_embedding.shape", image_emb.shape)
item_dict = {
    "key": image_emb_df["article_id"].cast(str),
    "value": image_emb
}
print("Save image_emb_dim64.npz...")
np.savez(f"./{dataset_version}/image_emb_dim64.npz", **item_dict)

contrast_emb_df = pl.read_parquet(contrast_emb_path)
contrast_emb = pca.fit_transform(np.array(contrast_emb_df["contrastive_vector"].to_list()))
print("contrast_emb.shape", contrast_emb.shape)
item_dict = {
    "key": contrast_emb_df["article_id"].cast(str),
    "value": contrast_emb
}
print("Save contrast_emb_dim64.npz...")
np.savez(f"./{dataset_version}/contrast_emb_dim64.npz", **item_dict)

print("All done.")
