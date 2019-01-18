"""
Matching communication network with email network in the MC3 dataset
"""
import dev.util as util
import matplotlib
import matplotlib.pyplot as plt
from model.GromovWassersteinLearning import GromovWassersteinLearning
import numpy as np
import pickle
from sklearn.manifold import TSNE
import torch


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize='xx-small')
    ax.set_yticklabels(row_labels, fontsize='xx-small')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-75, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=2)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0.15:
                kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts


def plot_results(gwl_model, index_s, index_t, epoch):
    # tsne
    embs_s = gwl_model.emb_model[0](index_s)
    embs_t = gwl_model.emb_model[1](index_t)
    embs = np.concatenate((embs_s.cpu().data.numpy(), embs_t.cpu().data.numpy()), axis=0)
    embs = TSNE(n_components=2).fit_transform(embs)
    plt.figure(figsize=(5, 5))
    plt.scatter(embs[:embs_s.size(0), 0], embs[:embs_s.size(0), 1],
                marker='x', s=10, c='b', edgecolors='b', label='Diseases')
    plt.scatter(embs[-embs_t.size(0):, 0], embs[-embs_t.size(0):, 1],
                marker='o', s=10, c='', edgecolors='r', label='Procedures')
    leg = plt.legend(loc='upper left', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel('T-SNE of node embeddings')
    plt.savefig('mimic3_epoch{}.pdf'.format(epoch))
    plt.close("all")


data_name = 'mimic3_2'
result_folder = 'match_mimic3_2'
cost_type = ['cosine']
method = ['proximal']

filename = '{}/result_{}_{}.pkl'.format(result_folder, method[0], cost_type[0])
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    result_mc3 = pickle.load(f)

filename = '{}/{}_database.pickle'.format(util.DATA_TRAIN_DIR, data_name)
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    data_mc3 = pickle.load(f)

disease = []
procedure = []
for key in data_mc3['src_index'].keys():
    # idx = data_mc3['src_index'][key]
    disease.append('d'+key)

for key in data_mc3['tar_index'].keys():
    # idx = data_mc3['tar_index'][key]
    procedure.append('p'+key)

for m in method:
    for c in cost_type:
        hyperpara_dict = {'src_number': len(data_mc3['src_index']),
                          'tar_number': len(data_mc3['tar_index']),
                          'dimension': 50,
                          'loss_type': 'L2',
                          'cost_type': c,
                          'ot_method': m}
        index_s = torch.LongTensor(list(range(hyperpara_dict['src_number'])))
        index_t = torch.LongTensor(list(range(hyperpara_dict['tar_number'])))

        gwd_model = GromovWassersteinLearning(hyperpara_dict)

        # load model
        gwd_model.load_model('{}/model_{}_{}.pt'.format(result_folder, m, c))
        cost_st = gwd_model.gwl_model.mutual_cost_mat(index_s, index_t).cpu().data.numpy().transpose()
        harvest = result_mc3[4].transpose()  # gwd_model.trans.transpose()

        for i in range(harvest.shape[1]):
            dis = disease[i][1:]
            dis = data_mc3['src_title'][dis]

            for j in range(harvest.shape[0]):
                pro = procedure[j][1:]
                pro = data_mc3['tar_title'][pro]
                if harvest[j, i] > 0.15:
                    print('ot={:.2f}, {}: {} --> {}: {}'.format(harvest[j, i], disease[i], dis, procedure[j], pro))

        fig, ax = plt.subplots()
        im, cbar = heatmap(harvest, procedure, disease, ax=ax,
                           cmap="Wistia", cbarlabel="The optimal transport from diseases to procedures")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")
        fig.tight_layout()
        plt.savefig('maps.pdf')

        fig, ax = plt.subplots()
        im, cbar = heatmap(cost_st, procedure, disease, ax=ax,
                           cmap="Wistia", cbarlabel="The optimal transport from diseases to procedures")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")
        fig.tight_layout()
        plt.savefig('maps2.pdf')





