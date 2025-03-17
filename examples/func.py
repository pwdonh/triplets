import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from triplets.models import Ratings, CollateTrials
from triplets.fit import design_matrix

def simulate_triplet_dataset(model, df_items, df_raters, num_trials, items_per_trial):

    stim_cols = ['stim_0','stim_1','stim_2']
    item_variables = design_matrix(df_items, 'index')
    rater_variables = design_matrix(df_raters, 'index')

    items = []
    for trial in range(num_trials):
        items.append(np.random.permutation(item_variables.index)[:items_per_trial+1])
    items = np.array(items)

    dfs_triplets = []

    selected = np.ones(num_trials).astype(int)*3
    triplets_per_trial = items_per_trial-2

    for rater in df_raters.index:
        for i_triplet in range(triplets_per_trial):

            if i_triplet==0:
                df_triplets = pd.DataFrame(
                    index=np.arange(0,num_trials*triplets_per_trial,triplets_per_trial),
                    columns=stim_cols,
                    data=items[:,:3]
                )
            df_triplets['trial_index'] = np.arange(df_triplets.shape[0])
            df_triplets['rater'] = rater
            df_triplets['last_selected'] = selected
            df_triplets['selected'] = 0
            df_triplets['i_triplet'] = i_triplet

            data = Ratings(item_variables, rater_variables, df_triplets)

            x, y, lengths = CollateTrials('cpu')([d for d in data])
            yhat = model.forward(x, lengths)
            selected = yhat.data.argmax(1).numpy()
            df_triplets['selected'] = selected

            dfs_triplets.append(df_triplets.copy())
            df_triplets = df_triplets.copy()
            df_triplets.index += 1

            for trial, index in enumerate(df_triplets.index):
                df_triplets.loc[index,f'stim_{selected[trial]}'] = items[trial,3+i_triplet]

    df_triplets = pd.concat(dfs_triplets)
    return df_triplets.sort_index().sort_values('rater').reset_index(drop=True)


def mouse_cursor(scale, offset, linewidth=1):
    vertices = np.array([
        [ 0.4434576, -0.859418 ],
        [ 0.3175   , -0.582082 ],
        [ 0.5820834, -0.582082 ],
        [ 0.       ,  0.       ],
        [ 0.       , -0.846666 ],
        [ 0.1852084, -0.661458 ],
        [ 0.2961588, -0.918062 ],
        [ 0.4434576, -0.859418 ],
        [ 0.4434576, -0.859418 ]
    ])
    vertices *= scale
    vertices += offset
    path = Path(vertices, 
        np.array([ 1,  2,  2,  2,  2,  2,  2,  2, 79]))
    return patches.PathPatch(path, facecolor='w', lw=linewidth)


def Square(position, size, rounding, **kwargs):
    size *= 1.5
    xy = (position[0]-size/2, position[1]-size/2)
    width = size
    height = size
    return mpl.patches.FancyBboxPatch(
        xy, width, height,
        boxstyle=mpl.patches.BoxStyle("Round",rounding_size=rounding),
        **kwargs
    )


def plot_triplet(ax,colors=['#1f77b4','#ff7f0e','#2ca02c'],
                 roundings=[0,0,0], sizes=[15,15,15],
                 selected=None, mouse=False, p=None,
                 border_colors=['k','k','k'], border_width_1=1):
    
    width = 350
    pad = 0.025*width
    pad = 5
    r = width/2-pad
    # circle_size_1 = 0.03611111111111111*width
    circle_size_1 = 13
    border_width_1 = border_width_1
    border_width_2 = 2
    # border_color_1 = 'k'
    border_color_2 = (0,0,0,0.35)

    rect = mpl.patches.FancyBboxPatch(
        [pad+r/2,pad], r, r/3,
        edgecolor='k', facecolor=(1,1,1,0),
        boxstyle=mpl.patches.BoxStyle("Round",rounding_size=10),
        linewidth=border_width_1
    )
    ax.add_patch(rect)
    positions = []
    for ii, (color, rounding, size, border_color_1) in enumerate(zip(colors,roundings,sizes,border_colors)):
        position = (pad+2*r/3 + r/3*ii, pad+r/6)
        plot_mouse = False
        if (selected is not None)&(ii==selected):
            # circle_size = circle_size_1+2
            linewidth = border_width_2
            edgecolor = border_color_2
            if mouse:
                patch = mouse_cursor(25, np.array(position), border_width_1/2)
                plot_mouse = True
        else:
            # circle_size = circle_size_1
            linewidth = border_width_1
            edgecolor = border_color_1
        # if p is not None:
        #     c_size = circle_size_1*p[ii]*3
        # else:
        #     c_size = circle_size_1
        circle1 = Square(
            position, size, 
            facecolor=color, rounding=rounding,
            edgecolor=edgecolor, linewidth=linewidth
        )
        ax.add_patch(circle1)
        if plot_mouse:
            ax.add_patch(patch)
        positions.append(position)

    ax.set_xlim([r/2,width-r/2])
    ax.set_ylim([0,r/3+pad*2])
    ax.set_aspect('equal')
    ax.axis(False)
    return positions


def plot_triplet_row(ax, triplet, df_items=None, mouse=False, transparent=False,
                     p=None):
    if df_items is None:
        colors = sns.color_palette(as_cmap=True)[:3]
    else:
        colors = df_items.loc[triplet[['stim_0','stim_1','stim_2']],'hex'].values
        roundings = df_items.loc[triplet[['stim_0','stim_1','stim_2']],'rounding']
        sizes = df_items.loc[triplet[['stim_0','stim_1','stim_2']],'size']#*1.5-5
    if mouse:
        selected = triplet['selected']
    else:
        selected = None
    border_colors = ['k','k','k']
    if transparent:
        ii = triplet['selected']
        rgba = mpl.colors.to_rgba(colors[ii])
        colors[ii] = mpl.colors.to_hex((rgba[0],rgba[1],rgba[2],.1),keep_alpha=True)
        # import ipdb; ipdb.set_trace()
        border_colors[ii] = (.8,.8,.8)

    return plot_triplet(
        ax, colors=colors, roundings=roundings, 
        selected=selected, mouse=mouse, sizes=sizes,
        border_colors=border_colors, p=p
    )

def string_figure(fig, df_items, columns):

    xlims = [-25,610]
    ylims = [-70,370]

    ax = fig.subplots()
    # axs = fig.subplots(1,5,gridspec_kw=dict(width_ratios=[12,1,1,1,1]))
    # ax = axs[0]
    for ii, index in enumerate(df_items.index):
        position = ((ii%6)*50,(7-ii//6)*50)
        color = df_items.loc[ii,'hex']
        item = Square(
            position=position, size=df_items.loc[ii,'size']*1.5-7,
            facecolor=color, 
            rounding=df_items.loc[ii,'rounding']
        )
        # ax.text(*position,str(ii))
        ax.add_patch(item)
        # ax.hlines(position[1],position[0],xlims[1],linestyle='-',color=(.5,.5,.5), linewidth=.4,zorder=-2)
    ax.set_aspect('equal')
    ax.axis(False)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks([])
    ax.set_yticks([])
    # adjust_axis(ax,0,0,.99,.99)
    # fig.tight_layout()