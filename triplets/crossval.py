from .fit import (
    get_model, evaluate,
    design_matrix, get_dataset_splits, 
    get_splits_rater, get_dataset_split    
)
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np
import torch
import os
from argparse import Namespace
import pickle
from time import time

def cross_validation_run(index, df_cv, args):

    start = time()

    # Get parameters for this cross validation run
    # print(df_cv.loc[index])
    args = Namespace(**df_cv.loc[index])

    df_items = pd.read_csv(args.csvfile_items, index_col=0)
    df_raters = pd.read_csv(args.csvfile_raters, index_col=0)
    df_triplets = pd.read_csv(args.csvfile_triplets, index_col=0)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Generate dataset splits
    item_variables = design_matrix(df_items, args.variables_item)
    rater_variables = design_matrix(df_raters, args.variables_rater)
    trainset, validset, testset, _, _, test_indices = get_dataset_splits(
        df_triplets, item_variables, rater_variables, args.fold, 
        args.batch_size, args.device, args.test_rater, args.seed
    )

    # == Fit on train set, test on test set
    model = get_model(trainset, validset, args, fit=True)
    results = evaluate(model, trainset, validset, testset, verbose=False)
    bits, corrects, rater_indices = model.test_individual(testset)
    results['bits'] = bits
    results['correct'] = corrects
    results['triplet'] = df_triplets.query(
        f'trial_index in {test_indices}'
    ).index.values.tolist()
    results['time'] = time()-start
    results['index'] = index
    return results

def expand_dataframe(df, column, values):
    num_rows = df.shape[0]
    num_values = len(values)
    df_new = pd.concat([df]*num_values)
    df_new.index = range(num_rows*num_values)
    state_dict_path = df['state_dict_path'].values[0]
    df_new['state_dict_path'] = [state_dict_path.replace('.pth', '_{}.pth'.format(index)) for index in df_new.index]
    df_new[column] = np.repeat(values, num_rows, 0)
    return df_new

def process_args(args):

    args_dict = vars(args)
    args_multiple = {key: args_dict[key] for key in args_dict.keys() if type(args_dict[key])==list}
    args_single = {key: args_dict[key] for key in args_dict.keys() if type(args_dict[key])!=list}

    df_cv = pd.DataFrame(args_single, index=[0])
    df_cv['valid_loss'] = None
    df_cv['valid_accuracy'] = None
    df_cv['test_loss'] = None
    df_cv['test_accuracy'] = None

    for key in args_multiple.keys():
#         if key=='test_rater':
#             raters = df_raters.index.values.tolist()
#             df_cv = expand_dataframe(df_cv, 'test_rater', raters)
#         else:
        df_cv = expand_dataframe(df_cv, key, args_multiple[key])

    return df_cv, args_multiple.keys()

def fit_triplets_crossval(args, csvpath=None):

    df_cv, multiple = process_args(args)
    # df_cv = expand_dataframe(df_cv, 'test_rater', [raters[0]])
    if csvpath is not None:
        df_cv_old = pd.read_csv(csvpath, index_col=0).fillna("")
        df_cv_old['cv_old'] = df_cv_old.index.values.astype(int)
        df_m = pd.merge(df_cv[multiple], df_cv_old[list(multiple)+['valid_loss']], how='left')
        rows_to_test = df_cv[pd.isna(df_m['valid_loss'])].index
        rows_to_copy = df_cv[~pd.isna(df_m['valid_loss'])].index
        df_cv_old.index = rows_to_copy
        df_cv.loc[rows_to_copy] = df_cv_old
    else:
        rows_to_test = df_cv.index

    indices_cv = []
    bits = []
    corrects = []
    indices_triplet = []
    if args.max_workers==1:

        for index in tqdm(rows_to_test):
            # print('mode: {}'.format(df_cv.loc[index,'mode']))
            results = cross_validation_run(index, df_cv, args)
            keys = np.setdiff1d(list(results.keys()), ['index','bits','correct','triplet'])
            for key in keys:
                df_cv.loc[results['index'], key] = results[key]
            indices_cv += [results['index']]*len(results['bits'])
            bits += results['bits']
            corrects += results['correct']
            indices_triplet += results['triplet']

    else:

        index = np.random.permutation(rows_to_test.values)

        # from acme import ParallelMap, esi_cluster_setup, cluster_cleanup
        # try:
        #     client = esi_cluster_setup(
        #         n_jobs=args.max_workers, 
        #         partition="8GBS", mem_per_job="2GB",
        #         # partition="E880", mem_per_worker="2GB",
        #         interactive=False, timeout=300
        #     )
        #     with ParallelMap(
        #         cross_validation_run, index, df_cv, args,
        #         write_worker_results=True, write_pickle=True
        #         # mem_per_job='2GB', partition='8GB', n_jobs=args.max_workers
        #     ) as pmap:
        #         filenames = pmap.compute() # returns a list of outputs
        # except KeyboardInterrupt:
        #     cluster_cleanup()
        # cluster_cleanup()

        # results = []
        # for fname in filenames:
        #     results.append(pickle.load(open(fname,'rb')))

        with Pool(args.max_workers) as pool:
            # cross_validation_run(index, df_cv, args)
            func = partial(cross_validation_run,
                df_cv=df_cv, args=args
                )
            results = list(tqdm(pool.imap(func, index), total=len(index)))

        for result in results:
            for key in np.setdiff1d(list(result.keys()), ['index','bits','correct','triplet']):
                df_cv.loc[result['index'], key] = result[key]
            indices_cv += [result['index']]*len(result['bits'])
            bits += result['bits']
            corrects += result['correct']            
            indices_triplet += result['triplet']

    df_individual = pd.DataFrame(dict(
        cv=indices_cv, triplet=indices_triplet, bits=bits, correct=corrects
    ))

    return df_cv, df_individual