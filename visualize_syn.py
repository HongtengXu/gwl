"""
Matching communication network with email network in the synthetic dataset
"""

import matplotlib.pyplot as plt
from model.GromovWassersteinLearning import GromovWassersteinLearning
import numpy as np
import pickle
from sklearn.manifold import TSNE
import torch


def plot_results(gwl_model, index_s, index_t, epoch):
    # tsne
    embs_s = gwl_model.emb_model[0](index_s)
    embs_t = gwl_model.emb_model[1](index_t)
    embs = np.concatenate((embs_s.cpu().data.numpy(), embs_t.cpu().data.numpy()), axis=0)
    embs = TSNE(n_components=2).fit_transform(embs)
    plt.figure(figsize=(5, 5))
    plt.scatter(embs[:embs_s.size(0), 0], embs[:embs_s.size(0), 1],
                marker='x', s=14, c='b', edgecolors='b', label='Email Net')
    plt.scatter(embs[-embs_t.size(0):, 0], embs[-embs_t.size(0):, 1],
                marker='o', s=12, c='', edgecolors='r', label='Call Net')
    leg = plt.legend(loc='upper left', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel('T-SNE of node embeddings')
    plt.savefig('emb2_epoch{}.pdf'.format(epoch))
    plt.close("all")


result_folder = 'match_syn'
cost_type = ['cosine']
method = ['proximal']
nl = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ntrail = 30
for n in [100, 50]:
    nc1 = np.zeros((2, 6, ntrail))
    nc2 = np.zeros((2, 6, ntrail))
    dgw = np.zeros((2, 6, ntrail))
    for i in range(6):

        for nn in range(ntrail):
            data_name = 'syn_{}_{}_{}'.format(nn, n, i)
            filename = '{}/result_{}_{}_{}.pkl'.format(result_folder, data_name, method[0], cost_type[0])
            with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
                result_mc3 = pickle.load(f)
            # GWD-OT
            nc1[0, i, nn] = result_mc3[0][0]
            # GWD-Emb
            nc2[0, i, nn] = result_mc3[2][0]
            # GWD-gw
            dgw[0, i, nn] = result_mc3[4][0]

            # GWL-c-OT
            nc1[1, i, nn] = result_mc3[0][-1]
            # GWL-c-Emb
            nc2[1, i, nn] = result_mc3[2][-1]
            # GWL-c-gw
            dgw[1, i, nn] = result_mc3[4][-1]

            # filename = '{}/result_{}_{}_{}.pkl'.format(result_folder, data_name, method[0], cost_type[1])
            # with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
            #     result_mc3 = pickle.load(f)
            #
            # # GWL-r-OT
            # nc1[2, i, nn] = result_mc3[0][-1]
            # # GWL-r-Emb
            # nc2[2, i, nn] = result_mc3[2][-1]
            # # GWL-r-gw
            # dgw[2, i, nn] = result_mc3[4][-1]

    y1m = np.mean(nc1, axis=2)
    y1s = np.std(nc1, axis=2)/3
    y2m = np.mean(nc2, axis=2)
    y2s = np.std(nc2, axis=2)/3
    y3m = np.mean(dgw, axis=2)
    y3s = np.std(dgw, axis=2)/3

    fig, ax1 = plt.subplots(1, 1, figsize=(5.7, 5), constrained_layout=True)
    ax2 = ax1.twinx()
    lns1 = ax1.errorbar(nl, y1m[0, :], y1s[0, :], color='black', capsize=3, linestyle='-.', label='GWD')
    lns1[-1][0].set_linestyle('-.')
    lns1 = ax1.errorbar(nl, y1m[1, :], y1s[1, :], capsize=3, label='GWL-C OT')
    lns1 = ax1.errorbar(nl, y2m[1, :], y2s[1, :], capsize=3, label='GWL-C Embedding')
    lns2 = ax2.errorbar(nl, y3m[1, :], y3s[1, :], capsize=3, color='g', label='GWL-C $d_{gw}$')

    ax1.set_xlabel('The percentage of noisy nodes and edges')
    ax1.set_ylabel('Node Correctness (%)')
    ax2.set_ylabel('GW discrepancy', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # tmp = lns1[2]
    lns = lns1+lns2
    for l in lns:
        print(l)
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower center')
    # plt.title('Convergence and Consistency')
    plt.savefig('syn_{}.pdf'.format(n))
    plt.close('all')



