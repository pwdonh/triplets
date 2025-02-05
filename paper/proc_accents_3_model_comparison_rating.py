
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import seaborn.objects as so
from argparse import Namespace

from proc_accents_1_model_comparison import model_comparison

# == Model comparison

config_path = 'configs/triplets3ab_crossval_rating.cfg'

hparams_fix = ['variables_item','variables_rater','fold']
hparams_optim = ['lambda_item','lambda_rater']

results_summary, cv_indices, _ = model_comparison(config_path, hparams_fix, hparams_optim)

# == Info partitions

df = pd.pivot(
    results_summary.query('variables_rater=="none"'),
    columns='variables_item',
    index='fold', values='test_loss'
).reset_index()
df['sex+group+rating_nativeness+index+rater'] = results_summary.query('variables_rater=="index"').query(
    'variables_item=="sex+group+rating_nativeness+index"'
)['test_loss'].values

df['full'] = df['none']-df['sex+group+rating_nativeness+index+rater']
df['raters_unique'] = df['sex+group+rating_nativeness+index']-df['sex+group+rating_nativeness+index+rater']
df['speaker_unique'] = df['sex+group+rating_nativeness']-df['sex+group+rating_nativeness+index']
df['sex_info'] = df['none']-df['sex']
df['model_nosex'] = df['sex']-df['sex+group+rating_nativeness']
df['rating_nosex'] = df['sex']-df['sex+rating_nativeness']
df['group_nosex'] = df['sex']-df['sex+group']
df['rating_unique'] = df['model_nosex']-df['group_nosex']
df['group_unique'] = df['model_nosex']-df['rating_nosex']
df['shared'] = df['model_nosex']-df['rating_unique']-df['group_unique']

# To long format
df_long = pd.melt(
    df, id_vars='fold', 
    value_vars=['sex_info','group_unique','shared','rating_unique','speaker_unique','raters_unique'],
    var_name='partition', value_name='loss'
)
resultdir = os.path.splitext(config_path)[0]
df_long.to_csv(
    os.path.join(resultdir, 'info_partitions.csv')
)

# == Unique info - individual raters

df_ind = pd.read_csv(
    os.path.join(resultdir, 'results_individual.csv'),
    index_col=0
)
df_results = pd.read_csv(
    os.path.join(resultdir, 'results.csv'),
    index_col=0
).fillna('none')
df_triplets = pd.read_csv(
    'data/triplets3ab_triplets.csv', index_col=0
)
df_rater = pd.read_csv(
    'data/triplets3ab_listeners.csv', index_col=0
)

# Select relevant CVs 
df = df_ind.query(f'cv in {cv_indices}').reset_index(drop=True).copy()
# Add CV info
df = pd.merge(df, df_results[hparams_fix], how='left', left_on='cv', right_index=True)
# Add rater
df = pd.merge(df, df_triplets['rater'], how='left', left_on='triplet', right_index=True)
# Add rater info
df = pd.merge(df, df_rater['group'], how='left', left_on='rater', right_index=True)
# To wide format
df_agg = df.groupby(['variables_item', 'variables_rater','rater'],as_index=False).agg({'bits':'mean','group':'first'})
df_wide = pd.pivot(
    df_agg.query('variables_rater=="index"'), index=['rater','group'], columns='variables_item', values='bits'
).reset_index()
df_wide['group_unique'] = df_wide['sex+rating_nativeness']-df_wide['sex+group+rating_nativeness']
df_wide['rating_unique'] = df_wide['sex+group']-df_wide['sex+group+rating_nativeness']
# Save
df_long = pd.melt(
    df_wide, id_vars=['rater','group'], 
    value_vars=['group_unique','rating_unique'],
    var_name='partition', value_name='bits'
)
df_long.to_csv(
    os.path.join(resultdir, 'uniques_individual.csv')
)