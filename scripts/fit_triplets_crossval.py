from triplets.fit import parse_args
from triplets.crossval import fit_triplets_crossval
import os

if __name__ == "__main__":

    args = parse_args()

    csvpath = os.path.join(args.output_folder, 'results.csv')
    csvpath_individual = os.path.join(args.output_folder, 'results_individual.csv')

    if os.path.exists(csvpath):
        df_cv, df_individual = fit_triplets_crossval(args, csvpath)
    else:
        df_cv, df_individual = fit_triplets_crossval(args)

    df_cv.to_csv(csvpath)
    df_individual.to_csv(csvpath_individual)