"""
Matching communication network with email network in the MC3 dataset
"""

import dev.util as util
import matplotlib.pyplot as plt
from model.GromovWassersteinLearning import GromovWassersteinLearning
import numpy as np
import pickle
import torch.optim as optim
from torch.optim import lr_scheduler


data_name = 'mimic3_2'
result_folder = 'match_mimic3_2'
cost_type = ['cosine']
method = ['proximal']

filename = '{}/{}_database.pickle'.format(util.DATA_TRAIN_DIR, data_name)
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    data_mc3 = pickle.load(f)

print(len(data_mc3['src_index']))
print(len(data_mc3['tar_index']))
print(len(data_mc3['src_interactions']))
print(len(data_mc3['tar_interactions']))

# connects = np.zeros((len(data_mc3['src_index']), len(data_mc3['tar_index'])))
# for item in data_mc3['src_interactions']:
#     connects[item[0], item[1]] += 1
# plt.imshow(connects)
# plt.savefig('{}/src.png'.format(result_folder))
# plt.close('all')
#
# connects = np.zeros((len(data_mc3['src_index']), len(data_mc3['tar_index'])))
# for item in data_mc3['tar_interactions']:
#     connects[item[0], item[1]] += 1
# plt.imshow(connects)
# plt.savefig('{}/tar.png'.format(result_folder))
# plt.close('all')

opt_dict = {'epochs': 5,
            'batch_size': 57000,
            'use_cuda': False,
            'strategy': 'soft',
            'beta': 1e-2,
            'outer_iteration': 200,
            'inner_iteration': 1,
            'sgd_iteration': 500,
            'prior': False,
            'prefix': result_folder,
            'display': False}

for m in method:
    for c in cost_type:
        hyperpara_dict = {'src_number': len(data_mc3['src_index']),
                          'tar_number': len(data_mc3['tar_index']),
                          'dimension': 50,
                          'loss_type': 'L2',
                          'cost_type': c,
                          'ot_method': m}

        gwd_model = GromovWassersteinLearning(hyperpara_dict)

        # initialize optimizer
        optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=1e-3)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        # Gromov-Wasserstein learning
        gwd_model.train_with_prior(data_mc3, optimizer, opt_dict, scheduler=None)
        # save model
        gwd_model.save_model('{}/model_{}_{}.pt'.format(result_folder, m, c))
        gwd_model.save_recommend('{}/result_{}_{}.pkl'.format(result_folder, m, c))



