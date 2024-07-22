import pickle
import sys
import os
import numpy as np
import Levenshtein

# wt_seq = 'NFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS'
wt_seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'

pkl_path = '/home/caiyi/low_n/outputs/pkl/'
exp_idx = sys.argv[1]


if exp_idx.isnumeric():
    sys.path.append('/home/caiyi/low_n/')
    import low_n_utils.A003_common

    file_list = os.listdir(pkl_path)
    sel_file_list = [f for f in file_list if f.startswith(exp_idx)]
    assert len(sel_file_list) == 1, 'Selected no or more than 1 pickle files!'
    filepath = os.path.join(pkl_path, sel_file_list[0])
    f = open(filepath, 'rb')
    filename = os.path.basename(filepath)
else:
    sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/A003_policy_optimization')
    sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/common')
    import A003_common

    f = open(exp_idx, 'rb')
    filename = os.path.basename(exp_idx)


a = pickle.load(f)

fit_his = a['sa_results']['fitness_history'][1:]
nmut_his = a['sa_results']['nmut_history'][1:]
accept_history = [[] for i in range(len(fit_his))]
delta_history = [[] for i in range(len(fit_his))]
delta_nmut_history = [[] for i in range(len(fit_his))]
last_iter = a['sa_results']['fitness_history'][0]
last_iter_nmut = a['sa_results']['nmut_history'][0]

for i, iter_ in enumerate(fit_his):
    for j in range(len(iter_)):
        if iter_[j] == last_iter[j]:
            accept_history[i].append(0)
            delta_history[i].append(0)
            delta_nmut_history[i].append(0)
        else:
            accept_history[i].append(1)
            delta_history[i].append(iter_[j] - last_iter[j])
            delta_nmut_history[i].append(nmut_his[i][j] - last_iter_nmut[j])
    last_iter = iter_
    last_iter_nmut = nmut_his[i]

accept_times_mean = np.mean(np.sum(accept_history, 0))
print('mean_accepct_times:', accept_times_mean)

delta_nonzeros = np.array(delta_history).ravel()[np.flatnonzero(delta_history)]
delta_mean = np.mean(delta_nonzeros)
# print(len(delta_nonzeros))
# print(len(np.flatnonzero(delta_history[0])))
print('mean_delta:', round(delta_mean, 4))

delta_nmut_nonzeros = np.array(delta_nmut_history).ravel()[np.flatnonzero(delta_nmut_history)]
delta_nmut_mean = np.mean(delta_nmut_nonzeros)
# print(len(delta_nmut_nonzeros))
# print(len(np.flatnonzero(delta_nmut_history[0])))
print('mean_delta_nmut:', round(delta_nmut_mean, 3))

with open('/home/caiyi/low_n/outputs/accept_summary.txt', 'a') as f:
    f.write(f'{exp_idx}\t{accept_times_mean}\t{delta_mean}\t{delta_nmut_mean}\n')
