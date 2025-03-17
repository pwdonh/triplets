import torch
import os
import numpy as np
import pandas as pd
from argparse import Namespace
import numpy as np
from shutil import copyfile
import argparse
import configparser
from .models import RaterModelTriplet, CollateTrials, Ratings
from sklearn.model_selection import StratifiedKFold
from formulaic import Formula

def get_default_args():
    return Namespace(
        num_dims=3, variables_item='index', variables_rater='index', distance='euclidean',
        fit_items=True, fit_raters=False, fit_sequence=False, force_positive=False,
        seed=123456, batch_size=50, reg_type=1, lambda_item=.005, lambda_rater=None,
        max_iter = 500, lr = .01, fold='5_0', test_rater=None, max_workers=1,
        verbose=False, device='cpu', state_dict_path='state_dict.pth'
    )

def parse_args(args=None, config_path=None):
    
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_path', type=str, default='configs/example.cfg')
        args = parser.parse_args()
    else:
        args = Namespace()

    if config_path is not None:
        args.config_path = config_path

    config = read_config(args.config_path)

    output_folder = config.config_path.replace('.cfg', '')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    copyfile(config.config_path, os.path.join(output_folder, 'config.cfg'))
    config.output_folder = output_folder
    config.state_dict_path = os.path.join(output_folder, 'model_state.pth')

    # copy values
    args = vars(args)
    config = vars(config)
    keys = np.setdiff1d(list(args.keys()), list(config.keys()))
    for key in keys:
        config[key] = args[key]

    return Namespace(**config)

def read_config(config_path):
    # Read configfile
    parser = configparser.ConfigParser()
    parser.read(config_path)
    # Read values
    config = {**dict(parser['data']),
              **dict(parser['training']),
              **dict(parser['model'])}
    vars_str = ['csvfile_items', 'csvfile_raters', 'csvfile_triplets',
                'experiment', 'distance', 'variables_item','force_positive',
                'variables_rater', 'test_rater', 'device', 'fold']
    vars_int = ['num_dims', 'seed', 'reg_type', 'batch_size', 'max_iter',
                'max_workers']
    vars_float = ['lambda_item', 'lambda_rater', 'lr']
    vars_bool = {'fit_items': 'model',
                 'fit_raters': 'model',    
                 'fit_sequence': 'model',
                 'verbose': 'training'}
    for key in config.keys():
        if config[key]=='None':
            config[key] = None
        elif config[key][0]=='[':
            config[key] = eval(config[key])
        else:
            if key in vars_str:
                config[key] = str(config[key])
            elif key in vars_int:
                config[key] = int(config[key])
            elif key in vars_float:
                config[key] = float(config[key])
            elif key in vars_bool.keys():
                config[key] = parser.getboolean(vars_bool[key], key)
    config['config_path'] = config_path
    return Namespace(**config)


def get_dataset_splits(df_triplets, item_variables, rater_variables, fold, batch_size, device, test_rater, seed):

    if test_rater is None:
        if fold!="none":
            train_indices, valid_indices, test_indices = get_splits_fold(df_triplets, fold, seed)
        else:
            train_indices = df_triplets['trial_index'].unique().tolist()
            valid_indices = df_triplets['trial_index'].unique().tolist()
            test_indices = df_triplets['trial_index'].unique().tolist()
    else:
        train_indices, valid_indices, test_indices = get_splits_rater(df_triplets, fold, test_rater, seed)

    trainset = get_dataset_split(
        df_triplets.query('trial_index == @train_indices'), item_variables, 
        rater_variables, int(batch_size), device, shuffle=True
    )
    validset = get_dataset_split(
        df_triplets.query('trial_index == @valid_indices'), item_variables, 
        rater_variables, int(batch_size), device, shuffle=False
    )
    testset = get_dataset_split(
        df_triplets.query('trial_index == @test_indices'), item_variables, 
        rater_variables, int(batch_size), device, shuffle=False
    )

    return trainset, validset, testset, train_indices, valid_indices, test_indices

def get_dataset_split(df_triplets, item_variables, rater_variables, batch_size, device='cpu', shuffle=True):
    dataset = Ratings(items=item_variables, raters=rater_variables, triplets=df_triplets)
    dataset.loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        collate_fn=CollateTrials(device), shuffle=shuffle
    )
    return dataset

def get_splits_fold(df_triplets, fold, seed=123456):
    df_trials = df_triplets.drop_duplicates(
        subset=["trial_index"]
        ).loc[:,['trial_index','rater']].set_index('trial_index')
    # Split dataset
    state = np.random.RandomState(seed)
    num_folds, i_fold = tuple([int(x) for x in fold.split('_')])
    splits = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed).split(
        np.zeros(df_trials.shape[0]),df_trials['rater']
    )
    for i, (trainvalidsplit, testsplit) in enumerate(splits):
        if i==i_fold:
            # permuted = state.permutation(trainvalidsplit)
            testset = testsplit
            # validset = permuted[:len(testsplit)]
            # trainset = permuted[len(testsplit):]
        elif i==((i_fold-1)%num_folds):
            validset = testsplit
    # num_trials = df_trials.shape[0]
    # num_per_fold = int(np.floor(num_trials/num_folds))
    # fold_sizes = [num_per_fold]*(num_folds-1)+[num_trials-num_per_fold*(num_folds-1)]
    # folds = [0]+np.cumsum(fold_sizes).tolist()
    # folds = np.array([folds[:-1], folds[1:]]).T
    # testset = np.arange(*folds[i_fold])
    # validset = np.arange(*folds[i_fold-1])
    # trainset = np.setdiff1d(
    #     np.arange(num_trials), testset.tolist()+validset.tolist()
    # )
    indices = df_trials.index.values
    trainset = np.setdiff1d(np.arange(len(indices)), np.union1d(testset,validset))
    # state.shuffle(indices)
    train_indices = indices[trainset].tolist()
    valid_indices = indices[validset].tolist()
    test_indices = indices[testset].tolist()
    return train_indices, valid_indices, test_indices

def get_splits_rater(df_triplets, fold, test_rater, seed=123456):
    df_trials = df_triplets.drop_duplicates(
        subset=["trial_index"]
    ).loc[:,['trial_index','rater']].set_index('trial_index')
    import ipdb; ipdb.set_trace()
    # Split trials from one rater as testset
    indices = df_trials.index.values.tolist()
    if isinstance(test_rater, str):
        test_rater = eval(test_rater)
    else:
        test_rater = test_rater
    test_indices = df_trials.query('rater=={}'.format(test_rater)).index.tolist()
    # Split rest into trainset and validset
    rest_indices = np.setdiff1d(indices, test_indices).tolist()
    state = np.random.RandomState(seed)
    state.shuffle(rest_indices)
    num_valid = int(len(indices)*p_valid)
    train_indices = rest_indices[:-num_valid]
    valid_indices = rest_indices[-num_valid:]
    return train_indices, valid_indices, test_indices


def get_model(trainset, validset, args, rater_mask=None, fit=True):

    fit_items = False
    fit_raters = False
    if (len(args.variables_item)>0)|(args.variables_item=="none"):
        fit_items = args.fit_items
    if (len(args.variables_rater)>0)|(args.variables_rater=="none"):
        fit_raters = args.fit_raters
    model = RaterModelTriplet(
        trainset.dim_items, trainset.dim_raters, 
        args.num_dims, args.lambda_item, args.lambda_rater, args.distance,
        args.force_positive, fit_items, fit_raters, args.fit_sequence, 
        args.reg_type, args.state_dict_path, args.device, rater_mask
    )
    if fit:
        model.update(trainset, lr=args.lr, max_iter=args.max_iter, validset=validset, verbose=args.verbose)

    val_loss, val_accuracy = model.test(validset)
    if args.verbose:
        print('Valid loss: {}, Valid accuracy: {}'.format(val_loss, val_accuracy))

    return model

def evaluate(model, trainset, validset=None, testset=None, verbose=True):
    results = {}
    for dataset, splitname in zip([trainset, validset, testset], ['train', 'valid', 'test']):
        if dataset is not None:
            loss, accuracy = model.test(dataset)
            results[splitname+'_loss'] = loss
            results[splitname+'_accuracy'] = accuracy
            if verbose:
                print(f'{splitname} loss: {loss}, {splitname} accuracy: {accuracy}')
    return results

def design_matrix(df, variables):
    if (len(variables)==0)|(variables=="none"):
        variables = 'index'
    if 'index' in variables.split('+'):
        num_items = len(df.index)
        num_digits = int(np.ceil(np.log10(num_items)))
        df['index'] = np.arange(len(df.index))
        df['index'] = df['index'].astype(str).str.zfill(num_digits)
    formula = Formula(variables+' - 1')
    design = formula.get_model_matrix(
        df,
        ensure_full_rank=False,
        na_action='ignore'
    )
    if 'Intercept' in design.columns:
        design = design.drop(columns='Intercept')
    return design.astype(float)

def design_matrix_old(df, variables):
    if (len(variables)==0)|(variables=="none"):
        variables = 'index'
    # created those with patsy before, need fully specified design matrix
    design = {}
    for variable in variables.split('+'):
        if variable=='index':
            num_items = len(df.index)
            num_digits = int(np.ceil(np.log10(num_items)))
            formatspec = f'%0{num_digits}d'
            column = pd.Series([formatspec %ii for ii, _ in enumerate(df.index)])
        else:
            column = df[variable]
        if column.dtype in ['int64', 'float64']:
            # print('Numerical: {}'.format(variable))
            design[variable] = column.values
            design[variable] = design[variable]/design[variable].max()
        else:     
            # print('Categorical: {}'.format(variable))
            factor = pd.Categorical(column).copy()
            for category in factor.categories:
                design['{}[T.{}]'.format(variable,category)] = np.array(factor==category, dtype=float)
    return pd.DataFrame(design, index=df.index)

def save_triplet_results(results, args):
    args_dict = vars(args)
    df_cv = pd.DataFrame(args_dict, index=[0])
    for key in results.keys():
        df_cv[key] = results[key]
    outpath = os.path.join(args.output_folder, 'results.csv')
    df_cv.to_csv(outpath)    
    return outpath

def fit_triplets(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_items = pd.read_csv(args.csvfile_items, index_col=0)
    df_raters = pd.read_csv(args.csvfile_raters, index_col=0)
    df_triplets = pd.read_csv(args.csvfile_triplets, index_col=0)

    item_variables = design_matrix(df_items, args.variables_item)
    rater_variables = design_matrix(df_raters, args.variables_rater)
    
    trainset, validset, testset, _, _, _ = get_dataset_splits(
        df_triplets, item_variables, rater_variables, args.fold, 
        args.batch_size, args.device, args.test_rater, args.seed
    )

    print('{} raters, {} trials, {} triplets'.format(
        df_triplets['rater'].unique().shape[0], 
        df_triplets['trial_index'].unique().shape[0], 
        df_triplets.shape[0])
    )
    model = get_model(trainset, validset, args)

    if args.fold=='none':
        return evaluate(model, trainset, None, None), model
    else:
        return evaluate(model, trainset, validset, testset), model