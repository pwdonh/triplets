
import numpy as np
import pandas as pd
import os

from params import config_path, embs

ratings = ['rating_nativeness', 'rating_gender', 'rating_usuk']

# == Merge spreadsheets

resultdir = os.path.splitext(config_path)[0]
df_items = pd.read_csv(os.path.join(resultdir, 'best/item_embeddings.csv'), index_col=0)
df_ratings = pd.read_csv('data/triplets3ab_speakers_ratings.csv', index_col=0)
df_items = pd.merge(df_items, df_ratings, how='left', left_index=True, right_index=True)

print(df_items[ratings+embs].corr().loc[embs,ratings].T.round(2))

df_long = pd.melt(
    df_items, id_vars=['username','group','rating_nativeness','rating_gender','rating_usuk','nativeness','sex','usuk'], 
    value_vars=[f'emb{ii}' for ii in range(8)], var_name='feature', value_name='score'
)
df_long.to_csv(os.path.join(resultdir, 'best/speaker_scores.csv'))

df_raters = pd.read_csv(os.path.join(resultdir, 'best/rater_embeddings.csv'), index_col=0)
df_long = pd.melt(
    df_raters, id_vars=['participant_id','group','nativeness','sex'], 
    value_vars=[f'emb{ii}' for ii in range(8)], var_name='feature', value_name='weight'
)
df_long.to_csv(os.path.join(resultdir, 'best/rater_weights.csv'))