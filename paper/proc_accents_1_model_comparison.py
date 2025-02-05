import numpy as np
import pandas as pd
import os
import configparser

from params import config_path

hparams_fix = ['variables_item','variables_rater','fold']
hparams_optim = ['lambda_item','lambda_rater']

# == Model comparison

def model_comparison(config_path, hparams_fix, hparams_optim):

    # Load results from cross-validation
    resultdir = os.path.splitext(config_path)[0]
    results = pd.read_csv(os.path.join(resultdir, 'results.csv'), index_col=0)
    results.fillna('none',inplace=True)

    # Find best hyperparameters per design and fold based on validation loss
    cv_indices = []
    for hparam in results[hparams_fix].drop_duplicates().values:
        # Find corresponding crossvals
        index = results.index.values
        for hp, hpname in zip(hparam, hparams_fix):
            index = index[results.loc[index,hpname]==hp]
        cv_indices += [results.loc[index,'valid_loss'].idxmin()]

    # Save corresponding test losses
    results_summary = results.loc[cv_indices].reset_index(drop=True)[hparams_fix+['test_loss','test_accuracy']]
    results_summary['test_loss'] = np.log2(np.exp(results_summary['test_loss']))

    # Hyperparameters for full model fit
    results_agg = results.groupby(
        hparams_fix+hparams_optim,as_index=False
    ).agg({'valid_loss':'mean','test_loss':'mean'})
    results_best = results_agg.groupby(hparams_fix).agg({'valid_loss':'idxmin'})
    df_hparams = results_agg.loc[results_best['valid_loss'].values,hparams_fix+hparams_optim]
    df_hparams = df_hparams.reset_index(drop=True)

    return results_summary, cv_indices, df_hparams

if __name__=='__main__':

    results_summary, cv_indices, df_hparams = model_comparison(config_path, hparams_fix, hparams_optim)

    resultdir = os.path.splitext(config_path)[0]
    results_summary.to_csv(
        os.path.join(resultdir, 'results_by_fold_log2.csv')
    )

    # == Info partitions

    df_wide = pd.pivot(
        results_summary.query('variables_rater=="none"'),
        columns='variables_item',
        index='fold', values='test_loss'
    ).reset_index()
    # Compute partitions
    df_wide['full'] = df_wide['none']-df_wide['group+sex+index']
    df_wide['demo'] = df_wide['none']-df_wide['group+sex'] # demographics, L1 + gender
    # Speaker: whatever is not captured by demographics
    df_wide['speaker_unique'] = df_wide['full']-df_wide['demo']
    # Demographics, uniques and shared
    df_wide['igroup'] = df_wide['none']-df_wide['group']
    df_wide['isex'] = df_wide['none']-df_wide['sex']
    df_wide['group_unique'] = df_wide['demo']-df_wide['isex'] # unique L1
    df_wide['sex_unique'] = df_wide['demo']-df_wide['igroup'] # unique gender
    df_wide['shared'] = df_wide['demo']-df_wide['group_unique']-df_wide['sex_unique']
    # shared is negative for some folds, distribute to unique infos
    df_wide['shared_leftover'] = df_wide['shared']
    df_wide.loc[df_wide['shared']>0,'shared_leftover'] = 0.
    df_wide.loc[df_wide['shared']<0,'shared'] = 0.
    df_wide['group_unique'] += df_wide['shared_leftover']/2
    df_wide['sex_unique'] += df_wide['shared_leftover']/2
    # Rater: whatever is added by including rater weights
    df_wide['raters'] = results_summary.loc[55:60]['test_loss'].values
    df_wide['raters_unique'] = df_wide['group+sex+index']-df_wide['raters']
    # To long format
    df_long = pd.melt(
        df_wide, id_vars='fold', value_vars=['sex_unique','shared','group_unique','speaker_unique','raters_unique'],
        var_name='partition', value_name='loss'
    )
    df_long.to_csv(
        os.path.join(resultdir, 'info_partitions.csv')
    )

    # == Save hyperparameters for full model (demographics + speaker) in a configuration file for fitting

    parser = configparser.ConfigParser()
    parser.read(config_path)

    for hparam in hparams_fix:
        parser['model'][hparam] = str(df_hparams.loc[4][hparam])
        # tag += parser['model'][hparam]+'_'
    for hparam in hparams_optim:
        parser['training'][hparam] = str(df_hparams.loc[4][hparam])        
    parser['training']['fold'] = 'none'

    config_path_best = os.path.join(resultdir,f'best.cfg')
    with open(config_path_best, 'w') as configfile:
        parser.write(configfile)