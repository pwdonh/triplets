# This reproduces the main figures in the paper, without any styling

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

from params import (
    speaker_groups, rater_groups, palette, embs
)

# == Model comparison & Info partitions

df = pd.read_csv('configs/triplets3ab_crossval/results_by_fold_log2.csv',index_col=0)

fig, axs = plt.subplots(1,2)
sns.lineplot(
    x='variables_item',y='test_loss',hue='fold',
    data=df.query('variables_rater=="none"'),
    ax=axs[0]
)
sns.lineplot(
    x='variables_rater',y='test_loss',hue='fold',
    data=df.query('variables_item=="group+sex+index"'),
    ax=axs[1]
)
fig.savefig('imgs/fig_4b.png')

df = pd.read_csv('configs/triplets3ab_crossval/info_partitions.csv',index_col=0)

fig, ax = plt.subplots()
p = (
    so.Plot(df, x='fold', y='loss', color='partition')
    .add(so.Bar(), so.Stack())
    .scale(x=so.Nominal(order=['5_4','5_3','5_2','5_1','5_0']))
    .on(ax)
)
p.plot()
fig.savefig('imgs/fig_4c.png')

# == Speaker scores & Rater weights

df_items = pd.read_csv('configs/triplets3ab_crossval/best/speaker_scores.csv', index_col=0)
df_raters = pd.read_csv('configs/triplets3ab_crossval/best/rater_weights.csv', index_col=0)

fig, axs = plt.subplots(2,1)
p = (
    so.Plot(df_items, x='feature', y='score')
    .add(so.Dots())
    .on(axs[0])
)
p.plot()
p = (
    so.Plot(df_raters, x='feature', y='weight')
    .add(so.Dots())
    .on(axs[1])
)
p.plot()
fig.savefig('imgs/fig_5a.png')

df_items = df_items.query(f'feature in {embs}')

fig = plt.figure()
p = (
    so.Plot(
        df_items, x='group', y='score', color='group', marker='sex'
    )
    .facet(col='feature')
    .add(so.Dots())
    .scale(
        x=so.Nominal(order=speaker_groups),
        color=so.Nominal(palette)
    )
    .on(fig)
)
p.plot()
fig.savefig('imgs/fig_5b.png')

df_raters = df_raters.query(f'feature in {embs}')

fig = plt.figure()
p = (
    so.Plot(
        df_raters, x='group', y='weight', color='group', marker='sex'
    )
    .facet(col='feature')
    .add(so.Dots())
.scale(
        x=so.Nominal(order=rater_groups),
        color=so.Nominal(palette)
    )
    .on(fig)
)
p.plot()
fig.savefig('imgs/fig_6cde_bottom.png')

# == Ratings vs. scores

fig = plt.figure()
p = (
    so.Plot(
        df_items.query('feature in ["emb0","emb1"]'), 
        x='rating_nativeness', y='score', color='nativeness'
    )
    .add(so.Dots())
    .add(so.Line(), so.PolyFit(order=1))
    .facet(col='feature')
    .on(fig)
)
p.plot()
fig.savefig('imgs/fig_6c_top.png')

fig = plt.figure()
p = (
    so.Plot(
        df_items.query('feature in ["emb2"]'), 
        x='rating_gender', y='score', color='sex'
    )
    .add(so.Dots())
    .add(so.Line(), so.PolyFit(order=1))
    .on(fig)
)
p.plot()
fig.savefig('imgs/fig_6d_top.png')

fig = plt.figure()
p = (
    so.Plot(
        df_items.query('feature in ["emb3","emb4"]'), 
        x='rating_usuk', y='score', color='usuk'
    )
    .add(so.Dots())
    .add(so.Line(), so.PolyFit(order=1))
    .facet(col='feature')
    .on(fig)
)
p.plot()
fig.savefig('imgs/fig_6e_top.png')

# == Model comparison updated + ratings

df = pd.read_csv('configs/triplets3ab_crossval_rating/info_partitions.csv',index_col=0)

fig, ax = plt.subplots()
p = (
    so.Plot(df, x='fold', y='loss', color='partition')
    .add(so.Bar(), so.Stack())
    .scale(x=so.Nominal(order=['5_4','5_3','5_2','5_1','5_0']))
    .on(ax)
)
p.plot()
fig.savefig('imgs/fig_7b.png')

df = pd.read_csv('configs/triplets3ab_crossval_rating/uniques_individual.csv',index_col=0)

fig, axs = plt.subplots(2,1)
for ax, partition in zip(axs, df['partition'].unique()):
    p = (
        so.Plot(df.query(f'partition=="{partition}"'), x='group', y='bits', color='group')
        .add(so.Dots(), so.Jitter())
        .add(so.Range(), so.Est(errorbar='ci'), so.Shift(.2))
        .scale(
            x=so.Nominal(order=rater_groups),
            color=so.Nominal(palette)
        )
        .on(ax)
    )
    p.plot()
    ax.hlines(0,*ax.get_xlim(),linestyle=':',color='#555555')
fig.savefig('imgs/fig_7c.png')

# == Asymmetry in selection probability

df = pd.read_csv('results/selection_probability.csv',index_col=0)

fig = plt.figure()
p = (
    so.Plot(df, x='nativeness',y='probability')
    .facet(col='group',order=['english_uk','german','french'])
    .add(so.Dots())
    .add(so.Range(),so.Est(),so.Shift(.2))
    .scale(x=so.Nominal(order=['hi','mid','lo']))
    .on(fig)
)
p.plot()
fig.savefig('imgs/fig_8b.png')