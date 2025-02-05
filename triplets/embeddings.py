from .fit import (
    get_model, design_matrix, get_dataset_splits
)
import torch
import numpy as np
import pandas as pd
import os
from argparse import Namespace

# is_pca = False

def reconstruct_embedding(linear, design, variables, do_umap=True):
    weights = linear.weight.data.numpy()
    index = [design.keys()=='Intercept']
#     index = []
    for factor in variables.split('+'):
        index.append([factor in key for key in design.keys()])
    index = np.any(index,0)
    weights = weights[:,index]
    embedding = np.dot(weights, design.iloc[:,index].values.T).T
    return embedding, weights

def pca_item_weights(model):
    U,S,Vh = torch.svd(model.linear_item.weight.data)
    num_dims = model.num_dims
    model.num_dims = num_dims
    model.linear_item.out_features = num_dims
    model.linear_rater.out_features = num_dims
    model.linear_item.weight.data = Vh.T[:num_dims].contiguous()
    num_rater = model.linear_rater.weight.data.shape[1]
    model.linear_rater.weight.data = S[:num_dims,None].repeat(1,num_rater)
    return model

def reconstruct(args, csvpath_cv=None):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_items = pd.read_csv(args.csvfile_items, index_col=0)
    df_raters = pd.read_csv(args.csvfile_raters, index_col=0)
    df_triplets = pd.read_csv(args.csvfile_triplets, index_col=0)

    if csvpath_cv is None:
        csvpath_cv = os.path.join(args.output_folder, 'results.csv')
    df_cv = pd.read_csv(csvpath_cv, index_col=0)

    item_variables = design_matrix(df_items, args.variables_item)
    rater_variables = design_matrix(df_raters, args.variables_rater)
    trainset, validset, _, _, _, _ = get_dataset_splits(
        df_triplets, item_variables, rater_variables, args.fold, 
        args.batch_size, args.device, args.test_rater, args.seed    
    )

    if (hasattr(args,'query_string')) and (len(args.query_string)>0):
        indices = df_cv.query(args.query_string).index
    else:
        indices = df_cv.index
    assert(len(indices)==1)

    args = Namespace(**df_cv.loc[indices[0]])
    model = get_model(trainset, validset, args, fit=False)
    model.load_state_dict(torch.load(args.state_dict_path))
    # if is_pca:
    #     model = pca_item_weights(model)

    # args.variables_item = 'group+index'
    emb_s = reconstruct_embedding(model.linear_item, item_variables, args.variables_item)[0]
    emb_r = reconstruct_embedding(model.linear_rater, rater_variables, args.variables_rater)[0]
    if args.force_positive=='sigmoid':
        emb_s = torch.sigmoid(torch.Tensor(emb_s)).numpy()
        emb_r = torch.nn.functional.softplus(torch.Tensor(emb_r)).numpy()
    elif args.force_positive=='softplus':
        emb_s = torch.nn.functional.softplus(torch.Tensor(emb_s)).numpy()        
        emb_r = torch.sigmoid(torch.Tensor(emb_r)).numpy()
    # emb_r = torch.nn.functional.softmax(torch.Tensor(emb_r), dim=1).numpy()

    order = np.flipud(np.argsort(np.linalg.norm(emb_s, axis=0)))
    emb_s = emb_s[:,order]
    emb_r = emb_r[:,order]

    # Count stimulus appearances
    if False:
        all_items = df_triplets.loc[:,['stim_0','stim_1','stim_2']].values
        items, counts = np.unique(all_items, return_counts=True)
        df_items.loc[items, 'count'] = counts
        df_items['count'] = df_items['count'].astype('int')

        all_raters = df_triplets['rater']
        raters, counts = np.unique(all_raters, return_counts=True)
        df_raters.loc[raters, 'count'] = counts
        df_raters['count'] = df_raters['count'].astype('int')

    columns = ['emb{}'.format(ii) for ii in range(emb_s.shape[1])]
    df_items.loc[:, columns] = emb_s
    df_raters.loc[:, columns] = emb_r
    df_raters['sequence_weight'] = torch.sigmoid(model.linear_sequence.weight.data[0])

    weights = model.linear_item.weight.data.numpy()
    # import ipdb; ipdb.set_trace()
    df_weights = pd.DataFrame(
        index=columns, columns=item_variables.columns,
        data=weights[order,:]
    )

    return df_items, df_raters, columns, df_weights
