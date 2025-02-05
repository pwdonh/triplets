import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns

df_triplets = pd.read_csv('data/triplets3ab_triplets.csv',index_col=0)
df_items = pd.read_csv('data/triplets3ab_speakers.csv',index_col=0)
df_ratings = pd.read_csv('data/triplets3ab_speakers_ratings.csv',index_col=0)
df_raters = pd.read_csv('data/triplets3ab_listeners.csv',index_col=0)

df_triplets[['lo','mid','hi']] = 0

for index in tqdm(df_triplets.index):
    triplet = df_triplets.loc[index]
    items = triplet[['stim_0','stim_1','stim_2']].values
    ratings = df_ratings.loc[items,'rating_nativeness'].values
    selected = np.zeros(3)
    selected[int(triplet['selected'])] = 1
    isorted = ratings.argsort()
    df_triplets.loc[index,['hi','mid','lo']] = selected[isorted]

df_grouped = df_triplets.groupby('rater',as_index=False).agg(
    {'lo':'mean','mid':'mean','hi':'mean'}
)
df_grouped = pd.merge(df_grouped, df_raters, how='left', left_on='rater', right_index=True)

df_long = pd.melt(
    df_grouped, id_vars=['rater','group'], 
    value_vars=['lo','mid','hi'],
    var_name='nativeness', value_name='probability'
)
df_long.to_csv('results/selection_probability.csv')