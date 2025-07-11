import pickle
import sys
import os
import Levenshtein
sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/A003_policy_optimization')
sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/common')
from low_n_utils import A003_common
sys.path.append('/home/caiyi/low_n/')
from low_n_utils import sequence_selection


wt_seq = 'MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS'
# wt_seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'

pkl_path = '/home/caiyi/low_n/outputs/pkl/'
# exp_idx = sys.argv[1]
algo = "ga"#sys.argv[2]
strategy = 'filter'
# strategy = 'last_iter'
BURNIN = 83
MAX_SA_ITR = None
nseq_select = 5500

from low_n_utils import low_n_utils 

# if exp_idx.isnumeric():
#     sys.path.append('/home/caiyi/low_n/')
#     import low_n_utils.A003_common
#     from low_n_utils import sequence_selection

#     file_list = os.listdir(pkl_path)
#     sel_file_list = [f for f in file_list if f.startswith(exp_idx)]
#     assert len(sel_file_list) == 1, 'Selected no or more than 1 pickle files!'
#     filepath = os.path.join(pkl_path, sel_file_list[0])
#     f = open(filepath, 'rb')
#     filename = os.path.basename(filepath)
# else:
#     sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/A003_policy_optimization')
#     sys.path.append('/home/caiyi/github/low-N-protein-engineering-master/analysis/common')
#     from low_n_utils import A003_common
#     sys.path.append('/home/caiyi/low_n/')
#     from low_n_utils import sequence_selection

#     f = open(f'outputs/pkl/{exp_idx}', 'rb')
#     filename = os.path.basename(exp_idx)


# algos = {'mcmc', 'ga'}
# for alg in algos.difference(set([algo])):
#     if alg in filename:
#         raise Exception(f'May be using the wrong method to parse the file {filename}!')
f = open('/home/wangqihan/Low_n_alphafold_test/design/220106_lin_pet_stability_ePtsRep_96_GA_0.p', "rb")#f'outputs/pkl/{exp_idx}', 'rb')
a = pickle.load(f)

print(a.keys())
print(a['sa_results'].keys())
print('num_chains:', len(a['sa_results']['seq_history'][0]))
print('iters:', len(a['sa_results']['seq_history']))

top_fitness = []
fitness_set = set()
seq_set = set()
dists = []

if algo == 'mcmc':
    history = []
    if strategy == 'best_iter':
        trajectories = [[] for _ in range(len(a['sa_results']['seq_history'][0]))]
        for i, iter_ in enumerate(a['sa_results']['fitness_history']):
            for j, fitness in enumerate(iter_):
                trajectories[j].append((i, fitness))
        max_ = []
        for traj in trajectories:
            max_.append(max(traj, key=lambda x:x[1]))

        for i, (iter_, fitness) in enumerate(max_):
            seq = a['sa_results']['seq_history'][iter_][i]
            if fitness not in fitness_set and seq not in seq_set:
                history.append((seq, fitness))
                fitness_set.add(fitness)
                seq_set.add(seq)

    elif strategy == 'last_iter':      
        for i in range(len(a['sa_results']['seq_history'][-1])):
            fitness = a['sa_results']['fitness_history'][-1][i]
            seq = a['sa_results']['seq_history'][-1][i]
            if fitness not in fitness_set and seq not in seq_set:
                history.append((seq, fitness))
                fitness_set.add(fitness)
                seq_set.add(seq)
    elif strategy == 'low_n_paper':
        select_df, res, res_sa = sequence_selection.select_top_seqs(f'outputs/pkl/{exp_idx}', nseq_select=nseq_select, burnin=BURNIN, max_sa_itr=MAX_SA_ITR)
        history = list(zip(select_df.seq, select_df.predicted_fitness))
    elif strategy == 'filter':
        select_df, res, res_sa = sequence_selection.filter_and_select_top_seqs(f'outputs/pkl/{exp_idx}', nseq_select=nseq_select, burnin=BURNIN, max_sa_itr=MAX_SA_ITR)
        
        vals = low_n_utils.validate_muts(wt_seq, list(select_df.seq))
        history = list(zip(select_df.seq, select_df.predicted_fitness, select_df.activity, vals))
    else:
        raise NameError('No such strategy!')

    history = sorted(history, key=lambda x:x[1], reverse=True)
    outfile_path = '/home/wangqihan/Low_n_alphafold_test/design/design_ga_seqs/eUniRep_pet_0_stability_ga_top3000.txt'#f'/home/wangqihan/Low_n_alphafold_test/design/design_ga_seqs/{filename[:-2]}_1000.txt'
    with open(outfile_path, 'w') as f:
        for seq, fitness, act, val in history[:nseq_select]:
            dist = Levenshtein.distance(seq, wt_seq)
            dists.append(dist)
            top_fitness.append(fitness)
            f.write(f'{seq}\t{fitness}\t{dist}\t{act}\t{val}\n')
        print(f'Writed to {outfile_path}')

elif algo == 'ga':
    fitness_history = []
    seq_history = []
    for i, iter_ in enumerate(a['sa_results']['seq_history']):
        seq_history.extend(iter_)
        fitness_history.extend(a['sa_results']['fitness_history'][i])
    history = {}
    for i in range(len(fitness_history)):
        if fitness_history[i] not in fitness_set:
            history[seq_history[i]] = fitness_history[i]
            fitness_set.add(fitness_history[i])

    
    history = sorted(list(history.items()), key=lambda x: x[1], reverse=True)
    outfile_path = '/home/wangqihan/Low_n_alphafold_test/design/design_ptsrep_ga_seqs/ePtsRep_pet_0_stability_ga_top3000.txt'#f'/home/caiyi/low_n/outputs/top_seqs/{filename[:-2]}_1000.txt'
    with open(outfile_path, 'w') as f:
        sel_seqs = []
        for seq, fit in history[:nseq_select]:
            sel_seqs.append(seq)
        # sel_activity = sequence_selection.predict_activity(sel_seqs, 'eptsrep_petase')
        # vals = low_n_utils.validate_muts(wt_seq, sel_seqs)
        for i, (seq, fit) in enumerate(history[:nseq_select]):
            dist = Levenshtein.distance(seq, wt_seq)
            dists.append(dist)
            top_fitness.append(fit)
            f.write(f'{seq}\n')
        print(f'Writed to {outfile_path}')

else:
    raise NameError('No such algorithm!')

print('Max predicted fitness:', round(max(top_fitness), 3))
print('Mean predicted fitness:', round(sum(top_fitness) / len(top_fitness), 3))
for i in range(max(dists) + 1):
    count = dists.count(i)
    print(f'n_mut={i}: {count}')
