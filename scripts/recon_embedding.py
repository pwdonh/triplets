import os
from triplets.fit import (
    parse_args, fit_triplets, save_triplet_results
)
from triplets.embeddings import reconstruct

if __name__ == "__main__":

    args = parse_args()

    print(args)

    outpath = os.path.join(args.output_folder, 'results.csv')

    df_items, df_raters, columns, df_weights = reconstruct(args, outpath)

    df_items.to_csv(os.path.join(args.output_folder, 'item_embeddings.csv'))
    df_raters.to_csv(os.path.join(args.output_folder, 'rater_embeddings.csv'))
    df_weights.to_csv(os.path.join(args.output_folder, 'item_weights.csv'))