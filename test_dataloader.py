"""
Build databases
"""

from preprocess.DataIO import build_dict_mimic3, build_dict_mc3, build_dict_ppi, build_dict_syn
import dev.util as util
import pickle

# syn
nl = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
for n_trail in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
    for num in [50, 100]:
        for i in range(len(nl)):
            data_syn = build_dict_syn(num_node=num,
                                      noise=nl[i])
            filename = '{}/syn_{}_{}_{}_database.pickle'.format(util.DATA_TRAIN_DIR, n_trail, num, i)
            with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(data_syn, f)
            print(len(data_syn['src_interactions']))
            print(len(data_syn['tar_interactions']))

# # ppi
# ppi_data_paths = ['{}/PPI_data/syn/0Krogan_2007_high.tab'.format(util.DATA_RAW_DIR),
#                   '{}/PPI_data/syn/0Krogan_2007_high+5e.tab'.format(util.DATA_RAW_DIR),
#                   '{}/PPI_data/syn/0Krogan_2007_high+10e.tab'.format(util.DATA_RAW_DIR),
#                   '{}/PPI_data/syn/0Krogan_2007_high+15e.tab'.format(util.DATA_RAW_DIR),
#                   '{}/PPI_data/syn/0Krogan_2007_high+20e.tab'.format(util.DATA_RAW_DIR),
#                   '{}/PPI_data/syn/0Krogan_2007_high+25e.tab'.format(util.DATA_RAW_DIR)]
# for i in [1, 2, 3, 4, 5]:
#     data_ppi = build_dict_ppi(ppi_data_paths[0], ppi_data_paths[i])
#     filename = '{}/ppi_{}_database.pickle'.format(util.DATA_TRAIN_DIR, i)
#     with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
#         pickle.dump(data_ppi, f)
#     print(len(data_ppi['src_interactions']))
#     print(len(data_ppi['tar_interactions']))

# # mc3
# data_paths = ['{}/MC3_data/meetings.csv'.format(util.DATA_RAW_DIR),
#               '{}/MC3_data/purchases.csv'.format(util.DATA_RAW_DIR),
#               '{}/MC3_data/emails.csv'.format(util.DATA_RAW_DIR),
#               '{}/MC3_data/calls.csv'.format(util.DATA_RAW_DIR)]
# with open('{}/MC3_data/node_subset.pickle'.format(util.DATA_RAW_DIR), 'rb') as f:  # Python 3: open(..., 'wb')
#     node2idx, pair2idx = pickle.load(f)
# data_mc3 = build_dict_mc3(data_paths[2], data_paths[3], node2idx)
# data_mc3_2 = build_dict_mc3(data_paths[2], data_paths[3], node2idx, pair2idx)
# filename = '{}/mc3_database.pickle'.format(util.DATA_TRAIN_DIR)
# with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(data_mc3, f)
# filename = '{}/mc3_2_database.pickle'.format(util.DATA_TRAIN_DIR)
# with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(data_mc3_2, f)
# print(len(data_mc3['src_interactions']))
# print(len(data_mc3['tar_interactions']))
# print(len(data_mc3_2['src_interactions']))
# print(len(data_mc3_2['tar_interactions']))
#
# # mimic3
# diagnose_dict_path = '{}/ICD-9_diagnose_data_mimic3/D_ICD_DIAGNOSES.csv'.format(util.DATA_RAW_DIR)
# procedure_dict_path = '{}/ICD-9_diagnose_data_mimic3/D_ICD_PROCEDURES.csv'.format(util.DATA_RAW_DIR)
# diagnose_adm_path = '{}/ICD-9_diagnose_data_mimic3/DIAGNOSES_ICD.csv'.format(util.DATA_RAW_DIR)
# procedure_adm_path = '{}/ICD-9_diagnose_data_mimic3/PROCEDURES_ICD.csv'.format(util.DATA_RAW_DIR)
# data_mimic3 = build_dict_mimic3(diagnose_dict_path,
#                                 diagnose_adm_path,
#                                 procedure_dict_path,
#                                 procedure_adm_path,
#                                 min_count=1000)
# filename = '{}/mimic3_database.pickle'.format(util.DATA_TRAIN_DIR)
# with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(data_mimic3, f)
# print(len(data_mimic3['src_index']))
# print(len(data_mimic3['src_title']))
# print(len(data_mimic3['tar_index']))
# print(len(data_mimic3['tar_title']))
# obs = data_mimic3['mutual_interactions']
# print(obs[0][0])
# print(obs[0][1])
# print(len(data_mimic3['mutual_interactions']))

# # mimic3_2
# diagnose_dict_path = '{}/ICD-9_diagnose_data_mimic3/D_ICD_DIAGNOSES.csv'.format(util.DATA_RAW_DIR)
# procedure_dict_path = '{}/ICD-9_diagnose_data_mimic3/D_ICD_PROCEDURES.csv'.format(util.DATA_RAW_DIR)
# diagnose_adm_path = '{}/ICD-9_diagnose_data_mimic3/DIAGNOSES_ICD.csv'.format(util.DATA_RAW_DIR)
# procedure_adm_path = '{}/ICD-9_diagnose_data_mimic3/PROCEDURES_ICD.csv'.format(util.DATA_RAW_DIR)
# data_mimic3 = build_dict_mimic3(diagnose_dict_path,
#                                 diagnose_adm_path,
#                                 procedure_dict_path,
#                                 procedure_adm_path,
#                                 min_count=2000)
# filename = '{}/mimic3_2_database.pickle'.format(util.DATA_TRAIN_DIR)
# with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(data_mimic3, f)
# print(len(data_mimic3['src_index']))
# print(len(data_mimic3['src_title']))
# print(len(data_mimic3['tar_index']))
# print(len(data_mimic3['tar_title']))
# obs = data_mimic3['mutual_interactions']
# print(obs[0][0])
# print(obs[0][1])
# print(len(data_mimic3['mutual_interactions']))
