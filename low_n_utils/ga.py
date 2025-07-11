import os
import sys
import warnings
import multiprocessing as mp
import random
import copy
sys.path.append('/home/wangqihan/github/openfold-main/Low-N/low_n_utils')
import numpy as np
import tensorflow as tf
import constants


# WARNING. This will work fine, but Surge is no longer using this path. 
# Surge has collected all the checkpoints at s3://efficient-protein-design/evotuning_checkpoints
#
## These are the set of weight paths Surge is currently using. 
# paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH
# paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH
# paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH
EVOTUNED_UNIREP_MODEL_WEIGHT_PATH = "./evotuned/unirep" 


GFP_LIB_REGION = [29, 110] # [inclusive, exclusive] - see A051a mlpe-gfp-pilot
BLAC_LIB_REGION = [132, 213] # [inclusive, exclusive] - see A051a mlpe-gfp-pilot


    
    
    
### SIMULATED ANNEALING ###

def acceptance_prob(f_proposal, f_current, k, T):
    ap = np.exp((f_proposal - f_current)/(k*T))
    ap[ap > 1] = 1
    return ap

def make_n_random_edits(seq, nedits, alphabet=constants.AA_ALPHABET_STANDARD_ORDER,
        min_pos=None, max_pos=None): ## Test
    """
    min_pos is inclusive. max_pos is exclusive
    """
    
    lseq = list(seq)
    lalphabet = list(alphabet)
    
    if min_pos is None:
        min_pos = 0
    
    if max_pos is None:
        max_pos = len(seq)
    
    # Create non-redundant list of positions to mutate.
    l = list(range(min_pos, max_pos))
    nedits = min(len(l), nedits)
    random.shuffle(l)
    pos_to_mutate = l[:nedits]    
    
    for i in range(nedits):
        pos = pos_to_mutate[i]     
        aa_to_choose_from = list(set(lalphabet) - set([seq[pos]]))
                        
        lseq[pos] = aa_to_choose_from[np.random.randint(len(aa_to_choose_from))]
        
    return "".join(lseq)

def propose_seqs(seqs, mu_muts_per_seq, min_pos=None, max_pos=None):
    
    mseqs = []
    for i,s in enumerate(seqs):
        n_edits = np.random.poisson(mu_muts_per_seq[i]-1) + 1
        mseqs.append(make_n_random_edits(s, n_edits, min_pos=min_pos, max_pos=max_pos)) 
        
    return mseqs

AA_LIST = ['A', 'C', 'D', 'E', 'F', 
           'G', 'H', 'I', 'K', 'L', 
           'M', 'N', 'P', 'Q', 'R', 
           'S', 'T', 'V', 'W', 'Y']


def mutation(child, mutation_rate=0.0035):
    if np.random.rand() < mutation_rate:
        mutate_point = np.random.randint(0, len(child))
        child[mutate_point] = AA_LIST[np.random.randint(0, 20)]


def crossover_and_mutation(pop, crossover_rate = 0.8):
    pop_size = len(pop)
    dna_size = len(pop[0])
    new_pop = []
    for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
        child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < crossover_rate:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(pop_size)]	#再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=dna_size)	#随机产生交叉的点
            child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
        mutation(child)	       #每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def relu(x):
    return np.maximum(0, x)


def select(pop, fitness):    # nature selection wrt pop's fitness
    fitness_pos = relu(fitness) + 1e-5
    pop_size = len(pop)
    idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True,
                           p=fitness_pos/(fitness_pos.sum()))
    print(idx, idx.dtype)
    return np.array(pop)[idx]

def anneal(
        init_seqs, 
        k, 
        T_max, 
        mu_muts_per_seq,
        get_fitness_fn,
        n_iter=1000, 
        decay_rate=0.99,
        min_mut_pos=None,
        max_mut_pos=None):
    
    print('Initializing')
    state_seqs = copy.deepcopy(init_seqs)
    state_fitness, state_fitness_std, state_fitness_mem_pred = get_fitness_fn(state_seqs)
    
    seq_history = [copy.deepcopy(state_seqs)]
    fitness_history = [copy.deepcopy(state_fitness)]
    fitness_std_history = [copy.deepcopy(state_fitness_std)]
    fitness_mem_pred_history = [copy.deepcopy(state_fitness_mem_pred)]


    for i in range(n_iter):
        print('Iteration:', i)
        
        # print('\tProposing sequences.')
        # proposal_seqs = propose_seqs(state_seqs, mu_muts_per_seq, 
        #         min_pos=min_mut_pos, max_pos=max_mut_pos)

        print('\tCrossover and mutation')
        seq_matrix = [list(seq) for seq in state_seqs]
        seq_matrix = crossover_and_mutation(seq_matrix)
        proposal_seqs = []
        for seq in seq_matrix:
            proposal_seqs.append(''.join(seq))
        
        print('\tCalculating predicted fitness.')
        proposal_fitness, proposal_fitness_std, proposal_fitness_mem_pred = get_fitness_fn(proposal_seqs)

        proposal_seqs = select(proposal_seqs, proposal_fitness)
        
        
        # print('\tMaking acceptance/rejection decisions.')
        # aprob = acceptance_prob(proposal_fitness, state_fitness, k, T_max*(decay_rate**i))
        
        # Make sequence acceptance/rejection decisions
        # for j, ap in enumerate(aprob):
            # if np.random.rand() < ap:
                # accept
                # state_seqs[j] = copy.deepcopy(proposal_seqs[j])
                # state_fitness[j] = copy.deepcopy(proposal_fitness[j])
                # state_fitness_std[j] = copy.deepcopy(proposal_fitness_std[j])
                # state_fitness_mem_pred[j] = copy.deepcopy(proposal_fitness_mem_pred[j])
            # else do nothing (reject)
        for j in range(len(proposal_seqs)):
            state_seqs[j] = copy.deepcopy(proposal_seqs[j])
            state_fitness[j] = copy.deepcopy(proposal_fitness[j])
            state_fitness_std[j] = copy.deepcopy(proposal_fitness_std[j])
            state_fitness_mem_pred[j] = copy.deepcopy(proposal_fitness_mem_pred[j])

        print('avg_state_fitness', np.mean(np.ma.masked_invalid(state_fitness)))

        seq_history.append(copy.deepcopy(state_seqs))
        fitness_history.append(copy.deepcopy(state_fitness))
        fitness_std_history.append(copy.deepcopy(state_fitness_std))
        fitness_mem_pred_history.append(copy.deepcopy(state_fitness_mem_pred))
        
    return {
        'seq_history': seq_history,
        'fitness_history': fitness_history,
        'fitness_std_history': fitness_std_history,
        'fitness_mem_pred_history': fitness_mem_pred_history,
        'init_seqs': init_seqs,
        'T_max': T_max,
        'mu_muts_per_seq': mu_muts_per_seq,
        'k': k,
        'n_iter': n_iter,
        'decay_rate': decay_rate,
        'min_mut_pos': min_mut_pos,
        'max_mut_pos': max_mut_pos,
    }
