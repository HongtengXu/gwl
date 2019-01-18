"""
Gromov-Wasserstein learning for ICD code
"""

import csv
import dev.util as util
import matplotlib.pyplot as plt
import numpy as np
import pickle


filename = '{}/mimic3_dicts.pickle'.format(util.DATA_TRAIN_DIR)
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    diag_dict, proc_dict = pickle.load(f)

filename = '{}/mimic3_data_tiny.pickle'.format(util.DATA_TRAIN_DIR)
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    diag2idx, proc2idx, Cost_d, Cost_p, mu_d, mu_p, Trans_train = pickle.load(f)

idx2diag = {}
for icd in diag2idx.keys():
    idx = diag2idx[icd]
    idx2diag[idx] = icd

idx2proc = {}
for icd in proc2idx.keys():
    idx = proc2idx[icd]
    idx2proc[idx] = icd

Cost_d /= np.max(Cost_d)
Cost_p /= np.max(Cost_p)

Beta = [1e-3, 1e-2, 1e-1, 1]
Js = [1, 10, 100]
iter = 2000
dgw = np.zeros((len(Beta), iter, 2))

mu_d2 = np.reshape(mu_d, [mu_d.shape[0], 1])
mu_p2 = np.reshape(mu_p, [mu_p.shape[0], 1])
Cost_dp = np.matmul(np.matmul(Cost_d**2, mu_d2), np.ones((1, mu_p.shape[0]))) + \
          np.matmul(np.matmul(np.ones((mu_d.shape[0], 1)), np.transpose(mu_p2)), np.transpose(Cost_p**2))


for i in range(len(Beta)):
    beta = Beta[i]
    for j in range(len(Js)):
        J = Js[j]

        # ipp
        print('proximal')
        Gamma = np.matmul(mu_d2, np.transpose(mu_p2))
        a = np.ones((mu_d.shape[0], 1))/mu_d.shape[0]
        for t in range(iter):
            C = Cost_dp - 2 * np.matmul(np.matmul(Cost_d, Gamma), np.transpose(Cost_p))
            K = np.exp(-C / beta) * Gamma
            for k in range(J):
                b = mu_p2 / np.matmul(np.transpose(K), a)
                a = mu_d2 / np.matmul(K, b)
            Gamma = np.matmul(np.matmul(np.diag(a[:, 0]), K), np.diag(b[:, 0]))
            L = Cost_dp - 2 * np.matmul(np.matmul(Cost_d, Gamma), np.transpose(Cost_p))
            dgw[i, t, 0] = np.sum(L * Gamma)

        # sinkhorn
        print('sinkhorn')
        Gamma = np.matmul(mu_d2, np.transpose(mu_p2))
        a = np.ones((mu_d.shape[0], 1))
        for t in range(iter):
            C = Cost_dp - 2*np.matmul(np.matmul(Cost_d, Gamma), np.transpose(Cost_p))
            K = np.exp(-C/beta)
            for k in range(J):
                b = mu_p2 / np.matmul(np.transpose(K), a)
                a = mu_d2 / np.matmul(K, b)
            Gamma = np.matmul(np.matmul(np.diag(a[:, 0]), K), np.diag(b[:, 0]))
            L = Cost_dp - 2 * np.matmul(np.matmul(Cost_d, Gamma), np.transpose(Cost_p))
            dgw[i, t, 1] = np.sum(L * Gamma)

        plt.figure(figsize=(6, 6))
        plt.plot(range(iter), dgw[i, :, 0], label='proximal')
        plt.plot(range(iter), dgw[i, :, 1], label='sinkhorn')
        plt.legend(loc='upper right', fontsize='xx-large')
        plt.xlabel('The number of inner iteration N', fontsize='xx-large')
        plt.ylabel('Gromov-Wasserstein discrepancy', fontsize='xx-large')
        # plt.title('$\gamma$={}'.format(beta))
        plt.savefig('compare_J{}_B{}.pdf'.format(J, beta))
        plt.close("all")



