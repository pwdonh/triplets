import numpy as np
import pandas as pd
import os
from triplets.fit import (
    parse_args, fit_triplets, 
    save_triplet_results
)
from triplets.embeddings import (
    reconstruct
)
from triplets.crossval import (
    fit_triplets_crossval
)

# == Triplet model fit

args = parse_args(config_path='test.cfg')

results, _ = fit_triplets(args)
outpath = save_triplet_results(results, args)
df_items, df_raters, columns, df_weights = reconstruct(args, outpath)

df_items.to_csv(os.path.join(args.output_folder, 'item_embeddings.csv'))
df_raters.to_csv(os.path.join(args.output_folder, 'rater_embeddings.csv'))
df_weights.to_csv(os.path.join(args.output_folder, 'item_weights.csv'))

for filename in ['item_embeddings','rater_embeddings','item_weights','results']:
    df_new = pd.read_csv(
        os.path.join(args.output_folder, f'{filename}.csv'), index_col=0
    )
    df_old = pd.read_csv(
        os.path.join(args.output_folder, f'{filename}_test.csv'), index_col=0
    )
    if not df_new.equals(df_old):
        if 'embeddings' in filename:
            deviation = np.max(np.abs((df_new[['emb0','emb1','emb2']]-df_old[['emb0','emb1','emb2']])).values)
        elif 'weights' in filename:
            deviation = np.max(np.abs(df_new-df_old).values)
        else:
            deviation = np.max(np.abs(df_new.iloc[0,-6:]-df_old.iloc[0,-6:]).values)
        print(f'Maximum result deviation ({filename}): {deviation}')

# == Cross-validation

args = parse_args(config_path='test_crossval.cfg')

df_cv, df_individual = fit_triplets_crossval(args)

csvpath = os.path.join(args.output_folder, 'results.csv')
csvpath_individual = os.path.join(args.output_folder, 'results_individual.csv')
df_cv.to_csv(csvpath)
df_individual.to_csv(csvpath_individual)

for filename in ['results','results_individual']:
    df_new = pd.read_csv(
        os.path.join(args.output_folder, f'{filename}.csv'), index_col=0
    )
    df_old = pd.read_csv(
        os.path.join(args.output_folder, f'{filename}_test.csv'), index_col=0
    )
    if filename=='results':
        df_new = df_new.drop(columns='time')
        df_old = df_old.drop(columns='time')
    if not df_new.equals(df_old):
        if filename=='results':
            cols = ['valid_loss','valid_accuracy','test_loss','test_accuracy','train_accuracy','train_loss']
            deviation = np.max(np.abs(df_new[cols]-df_old[cols]).values)
        else:
            deviation = np.max(np.abs(df_new['bits']-df_old['bits']).values)
        print(f'Maximum result deviation ({filename}): {deviation}')
